"""Experiment settings configured in MAIN_NOTEBOOK.ipynb."""
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product
import pandas as pd


@dataclass(frozen=True)
class Preference:
    risk_aversion: float
    esg_appetite: float

    @property
    def key(self):
        return f"A{self.risk_aversion:g}_B{self.esg_appetite:g}"


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    profile: str
    model_seeds: tuple
    training_steps: int
    baskets: dict
    checkpoint_frequency: int
    risk_aversions: tuple
    esg_appetites: tuple
    data_source: str
    saved_manifest: object
    refresh_wrds: bool
    number_of_assets: int
    first_stock_seed: int
    candidate_pool_size: int
    maximum_collection_attempts: int
    financial_start: str
    esg_start: str
    end_date: str
    minimum_past_observations: int
    maximum_score_age_days: int
    publication_lag_business_days: int
    periods: dict
    run_test: bool
    rebalance_days: int
    episode_days: int
    starting_value: float
    transaction_cost: float
    periods_per_year: int
    annual_risk_free_rate: float
    estimation_days: int
    momentum_days: tuple
    volatility_days: tuple
    correlation_days: int
    reward_scale: float
    ppo_overrides: dict
    hidden_size: int
    device: str
    cpu_threads: int
    keep_checkpoints: bool
    show_figures: bool
    # Data identifiers and numerical solver controls are recorded with every run.
    financial_table: str = "crsp.dsf_v2"
    sustainability_library: str = "sustainalytics_all"
    esg_table: str = "esgrr_focus"
    esg_field_id: int = 181110112399
    utility_max_iterations: int = 20_000
    utility_tolerance: float = 1e-9

    @property
    def preferences(self):
        return [Preference(float(a), float(b))
                for a, b in product(self.risk_aversions, self.esg_appetites)]

    def as_dict(self):
        def normalise(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: normalise(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalise(v) for v in value]
            return value
        return normalise(asdict(self))

    def validate(self):
        if self.data_source not in ("wrds", "saved"):
            raise ValueError("data_source must be wrds or saved.")
        if set(self.baskets) != {"training", "validation", "test"}:
            raise ValueError("Provide training, validation and test basket counts.")
        if any(n < 1 for n in self.baskets.values()):
            raise ValueError("Every role needs at least one basket.")
        if self.number_of_assets < 2 or self.training_steps < 1 or self.checkpoint_frequency < 1:
            raise ValueError("Use at least two assets and positive training/checkpoint lengths.")
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Provide distinct model seeds.")
        if len({p.key for p in self.preferences}) != len(self.preferences):
            raise ValueError("Preference combinations must be distinct.")
        if any(p.risk_aversion <= 0 or p.esg_appetite < 0 for p in self.preferences):
            raise ValueError("A must be positive and B nonnegative.")
        if self.rebalance_days < 1 or self.episode_days < self.rebalance_days:
            raise ValueError("Episode length must cover a positive rebalance interval.")
        if not 0 <= self.transaction_cost < 1 or self.estimation_days < 2:
            raise ValueError("Check transaction cost and estimation window.")
        if self.starting_value <= 0 or self.reward_scale <= 0 or self.cpu_threads < 1:
            raise ValueError("Starting value, reward scale and CPU thread count must be positive.")
        forbidden = {"policy", "env", "seed", "device", "policy_kwargs"} & set(self.ppo_overrides)
        if forbidden:
            raise ValueError(f"Use named experiment settings, not ppo_overrides, for {forbidden}.")
        for role, (first, last) in self.periods.items():
            if pd.Timestamp(first) >= pd.Timestamp(last):
                raise ValueError(f"Invalid {role} period.")
        if not (pd.Timestamp(self.periods['training'][1]) < pd.Timestamp(self.periods['validation'][0])
                and pd.Timestamp(self.periods['validation'][1]) < pd.Timestamp(self.periods['test'][0])):
            raise ValueError("Training, validation and test periods must be chronological and non-overlapping.")
        if pd.Timestamp(self.financial_start) > pd.Timestamp(self.esg_start):
            raise ValueError("Financial warm-up must precede ESG investment dates.")
        return self
