"""Past-only features, plain sample covariance and training-only standardisation.

At state date t, features and covariance use information through t. Its allocation
earns returns on t+1. Standardisation is fitted on the training role only.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Panel:
    name: str
    path: Path
    digest: str
    dates: pd.DatetimeIndex
    assets: list
    permnos: tuple
    entityids: tuple
    returns: np.ndarray
    scores: np.ndarray
    covariance: np.ndarray
    features: np.ndarray
    feature_names: list
    lookback: int


@dataclass
class DataRange:
    panel: Panel
    start: int
    stop: int
    role: str


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values):
        # Apply the means and standard deviations fitted on training observations.
        return (values - self.mean) / self.std


@dataclass
class Study:
    ranges: list
    scaler: Scaler
    files: dict


def load_panel(path, cfg):
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "permno", "ticker", "return", "overall_esg_risk"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing panel columns: {sorted(required-set(df.columns))}")
    if df.duplicated(["date", "permno"]).any():
        raise ValueError("Duplicate stock/date observations are not allowed.")
    df = df.sort_values(["date", "permno"])
    ids = tuple(sorted(int(x) for x in df.permno.unique()))
    if len(ids) != cfg.number_of_assets:
        raise ValueError("Basket size differs from number_of_assets.")
    r = df.pivot(index="date", columns="permno", values="return").reindex(columns=ids)
    q = df.pivot(index="date", columns="permno", values="overall_esg_risk").reindex(columns=ids)
    returns, scores = r.to_numpy(float), q.to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Incomplete financial returns or return <= -1. No dates were dropped.")
    if np.isinf(scores).any() or (scores[np.isfinite(scores)] < 0).any():
        raise ValueError("ESG risk must be nonnegative. Missing pre-ESG scores remain missing.")
    lookback = max(*cfg.momentum_days, *cfg.volatility_days, cfg.correlation_days, cfg.estimation_days)
    if len(r) <= lookback:
        raise ValueError("Insufficient financial warm-up for the configured feature windows.")
    names = (["latest_return"] + [f"momentum_{w}" for w in cfg.momentum_days]
             + [f"volatility_{w}" for w in cfg.volatility_days]
             + ["downside_volatility", "mean_asset_correlation", "relative_long_momentum", "overall_esg_risk"])
    f = np.zeros((len(r), len(ids), len(names)))
    covariance = np.zeros((len(r), len(ids), len(ids)))
    f[:, :, 0] = returns
    for t in range(len(r)):
        j, momenta = 1, []
        for window in cfg.momentum_days:
            momentum = np.expm1(np.log1p(returns[max(0, t-window+1):t+1]).sum(axis=0))
            f[t, :, j] = momentum
            momenta.append(momentum)
            j += 1
        for window in cfg.volatility_days:
            history = returns[max(0, t-window+1):t+1]
            f[t, :, j] = history.std(axis=0, ddof=1) if len(history) > 1 else 0
            j += 1
        history = returns[max(0, t-cfg.correlation_days+1):t+1]
        f[t, :, j] = np.sqrt(np.minimum(history, 0).__pow__(2).mean(axis=0))
        if len(history) > 1:
            sample = np.atleast_2d(np.cov(history, rowvar=False, ddof=1))
            std = np.sqrt(np.maximum(np.diag(sample), 0))
            denominator = np.outer(std, std)
            corr = np.divide(sample, denominator, out=np.zeros_like(sample), where=denominator > 0)
            np.fill_diagonal(corr, 0)
            f[t, :, j+1] = corr.sum(axis=1)/(len(ids)-1)
        long_momentum = momenta[int(np.argmax(cfg.momentum_days))]
        f[t, :, j+2] = long_momentum - long_momentum.mean()
        f[t, :, j+3] = scores[t]
        history = returns[max(0, t-cfg.estimation_days+1):t+1]
        if len(history) > 1:
            # Estimate sample covariance over the trailing financial window (ddof=1).
            covariance[t] = np.atleast_2d(np.cov(history, rowvar=False, ddof=1))
    last = df.drop_duplicates("permno", keep="last").set_index("permno")
    labels = [f"{last.loc[i, 'ticker']} [{i}]" for i in ids]
    entities = (tuple(int(last.loc[i, "sustainalytics_entityid"]) for i in ids)
                if "sustainalytics_entityid" in last.columns else ())
    digest = sha256(pd.util.hash_pandas_object(pd.concat([r, q], axis=1), index=True).values.tobytes()
                    + str(ids).encode()).hexdigest()
    return Panel(path.stem, path.resolve(), digest, r.index, labels, ids, entities,
                 returns, scores, covariance, f, names, lookback)


def checked_range(panel, role, cfg):
    first, last = map(pd.Timestamp, cfg.periods[role])
    first_index = int(panel.dates.searchsorted(first))
    start = first_index - 1           # Allocation is chosen BEFORE the first earned return.
    stop = int(panel.dates.searchsorted(last, side="right"))
    if start < panel.lookback-1 or stop-start-1 < 2:
        raise ValueError(f"{role}: insufficient warm-up or return observations.")
    if panel.dates[start+1] > first+pd.Timedelta(days=7) or panel.dates[stop-1] < last-pd.Timedelta(days=7):
        raise ValueError(f"{role}: basket does not cover the requested date boundaries.")
    if not np.isfinite(panel.features[start:stop-1]).all():
        raise ValueError(f"{role}: features/ESG scores unavailable or stale on an actionable date.")
    return DataRange(panel, start, stop, role)


def prepare_study(cfg, files):
    """Enforce stock separation and fit ONE scaler using training states only."""
    cfg.validate()
    files = {role: [(Path(p) if Path(p).is_absolute() else cfg.root/Path(p)).resolve()
                    for p in paths] for role, paths in files.items()}
    if any(not path.is_relative_to(cfg.root.resolve()) for paths in files.values() for path in paths):
        raise ValueError("Panels must be stored inside MAIN_NOTEBOOK_FOLDER for a self-contained run.")
    ranges, used_stocks, used_entities, seen = [], set(), set(), set()
    for role in ("training", "validation", "test"):
        if len(files[role]) != cfg.baskets[role]:
            raise ValueError(f"{role}: expected {cfg.baskets[role]} baskets.")
        role_calendar = None
        for path in files[role]:
            panel = load_panel(path, cfg)
            if panel.digest in seen or used_stocks.intersection(panel.permnos):
                raise ValueError("Baskets must contain distinct, non-overlapping stocks.")
            if used_entities.intersection(panel.entityids):
                raise ValueError("A rating entity appears in more than one basket.")
            seen.add(panel.digest)
            used_stocks.update(panel.permnos)
            used_entities.update(panel.entityids)
            checked = checked_range(panel, role, cfg)
            dates = panel.dates[checked.start:checked.stop]
            if role_calendar is not None and not dates.equals(role_calendar):
                raise ValueError(f"{role}: baskets have inconsistent return calendars. No dates were silently dropped.")
            role_calendar = dates
            ranges.append(checked)
    training = [r.panel.features[r.start:r.stop-1].reshape(-1, len(r.panel.feature_names))
                for r in ranges if r.role == "training"]
    values = np.concatenate(training)
    std = values.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # A constant input has no variation to standardise.
    return Study(ranges, Scaler(values.mean(axis=0), std), files)


def range_table(study, cfg):
    return pd.DataFrame([dict(role=r.role, basket=r.panel.name, basket_digest=r.panel.digest,
        path=str(r.panel.path.relative_to(cfg.root)), permnos=list(r.panel.permnos),
        first_return=str(r.panel.dates[r.start+1].date()), last_return=str(r.panel.dates[r.stop-1].date()),
        return_days=r.stop-r.start-1) for r in study.ranges])
