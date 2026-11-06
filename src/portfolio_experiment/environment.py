"""One accounting engine for PPO, static utility, rolling utility and equal weight."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .data import DataRange
from .finance import daily_reward


def action_to_weights(action):
    """[-1, 1] actions -> nonnegative amounts -> fully invested weights.

    Example: [1,-1,-1,-1] becomes [1,0,0,0]. If every amount is zero,
    return equal weights so that the portfolio remains fully invested.
    """
    action = np.asarray(action, dtype=float)
    if action.ndim != 1 or not np.isfinite(action).all():
        raise ValueError("An action must be a finite vector.")
    amounts = (np.clip(action, -1, 1)+1)/2
    total = amounts.sum()
    return amounts/total if total > 0 else np.full(len(action), 1/len(action))


class PortfolioEnv(gym.Env):
    """Observe through date t, trade before the next return, then hold to rebalance."""
    metadata = {"render_modes": []}

    def __init__(self, data_range, scaler, preference, cfg, shuffle=False):
        super().__init__()
        self.range, self.panel = data_range, data_range.panel
        self.scaler, self.preference, self.cfg = scaler, preference, cfg
        self.start, self.stop, self.shuffle = data_range.start, data_range.stop, shuffle
        self.n = len(self.panel.assets)
        self.order = np.arange(self.n)
        # Each stock row contains its market/ESG features and three portfolio-state inputs.
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(self.n, len(self.panel.feature_names)+3), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(self.n,), dtype=np.float32)

    def observation(self):
        market = self.scaler.transform(self.panel.features[self.day])
        # A terminal observation is not acted on. Retain the last available
        # feature row if an end-of-period ESG archive is unavailable there.
        if self.day == self.stop-1 and not np.isfinite(market).all():
            market = self.scaler.transform(self.panel.features[self.day-1])
        observation = np.concatenate([market, self.weights[:, None],
            np.full((self.n, 1), self.last_return), np.full((self.n, 1), self.last_turnover)], axis=1)
        observation = observation[self.order].astype(np.float32)
        if not np.isfinite(observation).all():
            raise ValueError("Nonfinite observation. Values were not silently clipped.")
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day, self.value = self.start, float(self.cfg.starting_value)
        self.weights = np.full(self.n, 1/self.n)
        self.last_return, self.last_turnover = 0., 0.
        self.order = self.np_random.permutation(self.n) if self.shuffle else np.arange(self.n)
        self.records = [dict(date=self.panel.dates[self.day], value=self.value, net_return=np.nan,
                             turnover=0., overall_esg_risk=np.nan,
                             **{f"weight:{a}": w for a, w in zip(self.panel.assets, self.weights)})]
        return self.observation(), {}

    def advance(self, target):
        """Apply target once, account daily, and let weights drift between trades."""
        target = np.asarray(target, dtype=float)
        if target.shape != (self.n,) or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1):
            raise ValueError("Portfolio weights must be nonnegative and sum to one.")
        reward_sum, count = 0., 0
        while count < self.cfg.rebalance_days and self.day < self.stop-1:
            p, cfg, t = self.panel, self.cfg, self.day
            held = target.copy() if count == 0 else self.weights.copy()
            turnover = .5*float(np.abs(held-self.weights).sum())
            gross = float(held@p.returns[t+1])
            cost = cfg.transaction_cost*turnover
            net = (1-cost)*(1+gross)-1
            reward = daily_reward(net, held, p.covariance[t], p.scores[t], self.preference, cfg)
            self.value *= 1+net
            self.weights = held*(1+p.returns[t+1])/(1+gross)
            self.last_return, self.last_turnover = net, turnover
            self.day += 1
            count += 1
            self.records.append(dict(date=p.dates[self.day], value=self.value, net_return=net,
                gross_return=gross, turnover=turnover, transaction_cost_fraction=cost,
                overall_esg_risk=float(held@p.scores[t]), equal_weight_esg_risk=float(p.scores[t].mean()),
                scheduled_rebalance=count == 1, **reward,
                **{f"weight:{a}": float(w) for a, w in zip(p.assets, held)}))
            reward_sum += reward["reward"]
        done = self.day == self.stop-1
        return self.observation(), float(reward_sum), done, False, {"market_days": count}

    def step(self, action):
        canonical = np.empty(self.n)
        canonical[self.order] = np.asarray(action)
        return self.advance(action_to_weights(canonical))

    def history(self):
        return pd.DataFrame(self.records).set_index("date")


class TrainingEnv(gym.Env):
    """One shared policy learns from uniformly sampled baskets and date blocks."""
    def __init__(self, ranges, scaler, preference, cfg):
        super().__init__()
        self.ranges, self.scaler, self.preference, self.cfg = ranges, scaler, preference, cfg
        self.env = PortfolioEnv(ranges[0], scaler, preference, cfg)
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        source = self.ranges[int(self.np_random.integers(len(self.ranges)))]
        length = min(self.cfg.episode_days, source.stop-source.start-1)
        begin = int(self.np_random.integers(source.start, source.stop-length))
        self.source_stop = source.stop
        block = DataRange(source.panel, begin, begin+length+1, "training_episode")
        self.env = PortfolioEnv(block, self.scaler, self.preference, self.cfg, shuffle=True)
        return self.env.reset(seed=int(self.np_random.integers(2**31-1)))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and self.env.stop < self.source_stop:
            terminated, truncated = False, True
        return obs, reward, terminated, truncated, info
