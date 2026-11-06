"""Expected utility controls, daily PPO reward and numerical optimisation."""
import numpy as np


def utility(weights, mean_annual, covariance_annual, raw_scores, a, b):
    """Expected annual return - A/2 * annual variance - B * raw ESG risk."""
    return float(weights @ mean_annual - a/2 * (weights @ covariance_annual @ weights)
                 - b * (weights @ raw_scores))


def daily_reward(net_return, weights, covariance_daily, raw_scores, preference, cfg):
    """Combine net return, estimated variance and relative ESG risk each day.

    The configured scale multiplies all three terms. The ESG term uses raw
    Sustainalytics points, with the basket's equal-weight score as its reference.
    Feature standardisation affects observations, not these reward calculations.
    """
    predicted_variance = float(weights @ covariance_daily @ weights)
    financial = float(net_return)
    risk = -preference.risk_aversion/2 * predicted_variance
    esg = (preference.esg_appetite/cfg.periods_per_year
           * (raw_scores.mean() - weights @ raw_scores))
    return dict(reward=cfg.reward_scale*(financial+risk+esg),
                reward_financial=cfg.reward_scale*financial,
                reward_risk=cfg.reward_scale*risk, reward_esg=cfg.reward_scale*esg,
                predicted_variance=predicted_variance)


def project_simplex(values):
    """Project candidate allocations onto non-negative weights summing to one."""
    ordered = np.sort(values)[::-1]
    excess = np.cumsum(ordered)-1
    active = np.flatnonzero(ordered-excess/np.arange(1, len(values)+1) > 0)[-1]
    return np.maximum(values-excess[active]/(active+1), 0)


def maximise_utility(mean_annual, covariance_annual, raw_scores, a, b,
                     maximum_iterations=20_000, tolerance=1e-9):
    """Maximise utility over non-negative weights summing to one.

    Projected gradient ascent uses the largest covariance eigenvalue to set
    its step size. Stop when the maximum weight change is below the supplied
    tolerance, or raise an error if the iteration limit is reached.
    """
    mean, cov, q = map(lambda x: np.asarray(x, dtype=float),
                       (mean_annual, covariance_annual, raw_scores))
    if not (np.isfinite(mean).all() and np.isfinite(cov).all() and np.isfinite(q).all()):
        raise ValueError("Utility inputs must be finite.")
    intercept = mean-b*q
    curvature = a*float(np.linalg.eigvalsh(cov).max())
    if curvature <= 0:
        w = np.zeros(len(mean))
        w[np.argmax(intercept)] = 1
        return w
    w = np.full(len(mean), 1/len(mean))
    for _ in range(maximum_iterations):
        next_w = project_simplex(w+(intercept-a*cov@w)/curvature)
        if np.max(np.abs(next_w-w)) < tolerance:
            return next_w/next_w.sum()
        w = next_w
    raise RuntimeError("Utility optimisation did not converge. No control was substituted.")


def control_target(panel, day, preference, cfg, financial_only=False):
    first = day-cfg.estimation_days+1
    history = panel.returns[first:day+1]
    b = 0.0 if financial_only else preference.esg_appetite
    mean = cfg.periods_per_year*history.mean(axis=0)
    covariance = cfg.periods_per_year*panel.covariance[day]
    return maximise_utility(mean, covariance, panel.scores[day], preference.risk_aversion, b,
                            maximum_iterations=cfg.utility_max_iterations, tolerance=cfg.utility_tolerance)


def control_details(data_range, preference, cfg):
    """Save the exact first-date inputs and objective terms for manual inspection."""
    p, day = data_range.panel, data_range.start
    history = p.returns[day-cfg.estimation_days+1:day+1]
    mean = cfg.periods_per_year*history.mean(axis=0)
    cov = cfg.periods_per_year*p.covariance[day]
    w = control_target(p, day, preference, cfg)
    return dict(role=data_range.role, basket=p.name, preference=preference.key,
                asof=str(p.dates[day].date()), first_history=str(p.dates[day-cfg.estimation_days+1].date()),
                observations=len(history), assets=p.assets, target_weights=w.tolist(),
                expected_return_annual=mean.tolist(), sample_covariance_annual=cov.tolist(),
                raw_esg_scores=p.scores[day].tolist(),
                return_contribution=float(w@mean), variance_penalty=float(preference.risk_aversion/2*(w@cov@w)),
                esg_penalty=float(preference.esg_appetite*(w@p.scores[day])),
                utility=utility(w, mean, cov, p.scores[day], preference.risk_aversion, preference.esg_appetite))
