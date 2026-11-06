"""Evaluate saved PPO policies and financial/ESG utility controls.

All methods share portfolio accounting. Metrics and paired differences are
calculated per complete backtest before averaging seeds and baskets.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .environment import PortfolioEnv
from .finance import control_target, control_details

METRICS = ["total_return", "sharpe_ratio", "preference_utility", "score_risk",
           "volatility", "annual_variance", "max_drawdown", "average_turnover"]
PAIR_KEYS = ["role", "basket_digest", "risk_aversion", "esg_appetite"]


def run_policy(model, data_range, scaler, preference, cfg):
    env = PortfolioEnv(data_range, scaler, preference, cfg)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.history()


def run_control(data_range, scaler, preference, cfg, strategy):
    env = PortfolioEnv(data_range, scaler, preference, cfg)
    env.reset()
    static = strategy.endswith("static")
    financial_only = strategy.startswith("financial_")
    fixed = control_target(env.panel, env.start, preference, cfg, financial_only) if static else None
    while env.day < env.stop-1:
        if strategy == "equal_weight":
            target = np.full(env.n, 1/env.n)
        elif static:
            target = fixed
        else:
            target = control_target(env.panel, env.day, preference, cfg, financial_only)
        env.advance(target)
    return env.history()


def metrics(history, preference, cfg):
    daily = history.iloc[1:]
    r = daily.net_return.to_numpy(float)
    value = history.value.to_numpy(float)
    k, variance = cfg.periods_per_year, float(r.var(ddof=1))
    rf = (1+cfg.annual_risk_free_rate)**(1/k)-1
    esg = float(daily.overall_esg_risk.mean())
    return dict(final_value=float(value[-1]), total_return=float(value[-1]/value[0]-1),
        annual_arithmetic_return=float(k*r.mean()),
        annual_variance=k*variance, volatility=float(np.sqrt(k*variance)),
        sharpe_ratio=float(np.sqrt(k)*(r.mean()-rf)/np.sqrt(variance)) if variance > 0 else np.nan,
        max_drawdown=float(-(value/np.maximum.accumulate(value)-1).min()),
        score_risk=esg, average_turnover=float(daily.turnover.mean()),
        preference_utility=float(k*r.mean()-preference.risk_aversion/2*k*variance-preference.esg_appetite*esg),
        reward_financial=float(daily.reward_financial.mean()), reward_risk=float(daily.reward_risk.mean()),
        reward_esg=float(daily.reward_esg.mean()), return_days=len(r),
        rebalance_decisions=int(daily.scheduled_rebalance.sum()))


def paired_tables(results):
    selected = results[results.strategy.eq("ppo_selected")]
    frames = []
    for reference in ["ppo_initial", "ppo_final", "utility_rolling", "utility_static", "equal_weight"]:
        right = results[results.strategy.eq(reference)]
        keys = PAIR_KEYS+(["seed"] if reference.startswith("ppo_") else [])
        joined = selected.merge(right[keys+METRICS], on=keys, suffixes=("", "_reference"),
                                how="left", validate="one_to_one" if reference.startswith("ppo_") else "many_to_one",
                                indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError(f"Missing paired reference: {reference}")
        joined = joined.drop(columns="_merge")
        for metric in METRICS:
            joined["difference_"+metric] = joined[metric]-joined[metric+"_reference"]
        frames.append(joined.assign(reference=reference))
    return pd.concat(frames, ignore_index=True)


def summarise(results):
    """Average seeds within a basket, THEN average baskets equally."""
    columns = [*METRICS, "final_value", "annual_arithmetic_return"]
    basket = results.groupby(PAIR_KEYS+["strategy"], dropna=False)[columns].mean().reset_index()
    return basket.groupby(["role", "risk_aversion", "esg_appetite", "strategy"])[columns].mean().reset_index()


def evaluate(cfg, study, run_folder):
    """Read saved policies and write full daily records without holding all paths in RAM."""
    from stable_baselines3 import PPO
    run_folder = Path(run_folder)
    trajectories = run_folder/"trajectories"
    trajectories.mkdir(exist_ok=True)
    rows, allocations, explanations, curves = [], [], [], {}
    ranges = [r for r in study.ranges if cfg.run_test or r.role != "test"]

    def store(history, r, preference, strategy, seed=np.nan):
        if not strategy.startswith("ppo_"):
            # Re-score cached controls under the currently reported A/B. Their
            # allocation objective can still be financial-only (B=0).
            history = history.copy()
            history["reward_risk"] = -cfg.reward_scale*preference.risk_aversion/2*history.predicted_variance
            history["reward_esg"] = cfg.reward_scale*preference.esg_appetite/cfg.periods_per_year*(
                history.equal_weight_esg_risk-history.overall_esg_risk)
            history["reward"] = history.reward_financial+history.reward_risk+history.reward_esg
        seed_label = f"_seed{int(seed)}" if pd.notna(seed) else ""
        filename = f"{r.role}_{r.panel.digest[:12]}_{preference.key}_{strategy}{seed_label}.csv"
        path = trajectories/filename
        history.to_csv(path)
        common = dict(role=r.role, basket=r.panel.name, basket_digest=r.panel.digest,
                      preference=preference.key, risk_aversion=preference.risk_aversion,
                      esg_appetite=preference.esg_appetite, strategy=strategy, seed=seed)
        rows.append(dict(common, trajectory=str(path.relative_to(run_folder)), **metrics(history, preference, cfg)))
        key = (r.role, preference.risk_aversion, preference.esg_appetite, strategy,
               int(seed) if pd.notna(seed) else "control")
        series = history.value
        if key not in curves:
            curves[key] = [series.copy(), 1]
        else:
            if not curves[key][0].index.equals(series.index):
                raise ValueError("Curve aggregation requires identical dates within each role.")
            curves[key][0] += series
            curves[key][1] += 1
        weights = history.iloc[1:].filter(like="weight:")
        for asset in weights:
            allocations.append(dict(common, asset=asset[7:], average_weight=weights[asset].mean(),
                minimum_weight=weights[asset].min(), maximum_weight=weights[asset].max()))

    for r in ranges:
        cache = {}
        for preference in cfg.preferences:
            explanations.append(control_details(r, preference, cfg))
            for strategy in ["utility_static", "utility_rolling", "financial_static", "financial_rolling", "equal_weight"]:
                # Financial and equal-weight controls need not be recalculated for every B.
                key = (strategy, preference.risk_aversion if strategy != "equal_weight" else 0,
                       preference.esg_appetite if strategy.startswith("utility_") else 0)
                if key not in cache:
                    cache[key] = run_control(r, study.scaler, preference, cfg, strategy)
                store(cache[key], r, preference, strategy)
    for preference in cfg.preferences:
        for seed in cfg.model_seeds:
            folder = run_folder/"models"/preference.key/f"seed{seed}"
            for strategy, name in [("ppo_initial", "initial.zip"), ("ppo_selected", "selected.zip"), ("ppo_final", "final.zip")]:
                model = PPO.load(folder/name, device=cfg.device)
                for r in ranges:
                    store(run_policy(model, r, study.scaler, preference, cfg), r, preference, strategy, int(seed))
                del model
            print(f"Evaluated {preference.key}, seed {seed} on complete saved periods.")
            # Persist completed results so a late interruption does not erase all evaluation work.
            pd.DataFrame(rows).to_csv(run_folder/"all_metrics.csv", index=False)
    results = pd.DataFrame(rows)
    pd.DataFrame(allocations).to_csv(run_folder/"allocation_summary.csv", index=False)
    (run_folder/"control_calculations.json").write_text(json.dumps(explanations, indent=2))
    paired_tables(results).to_csv(run_folder/"paired_comparisons.csv", index=False)
    summarise(results).to_csv(run_folder/"summary.csv", index=False)
    curve_rows = []
    for (role, a, b, strategy, seed), (series, count) in curves.items():
        curve_rows.append(pd.DataFrame(dict(date=series.index, value=series.to_numpy()/count,
            role=role, risk_aversion=a, esg_appetite=b, strategy=strategy,
            seed=np.nan if seed == "control" else seed, baskets=count)))
    pd.concat(curve_rows, ignore_index=True).to_csv(run_folder/"value_curves.csv", index=False)
    return results
