"""Fit the shared PPO policy and select checkpoints using validation utility.

The model uses SB3 defaults together with any explicit configuration overrides.
Training episodes, checkpoint scores and the complete run protocol are saved
so that fitted policies can be evaluated and inspected later.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import inspect
import json
import platform
import random
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .data import range_table
from .environment import TrainingEnv
from .evaluation import run_policy, metrics
from .policy import SharedStockPolicy


class ValidationSelection(BaseCallback):
    """Evaluate only after updates. Never use the test role to select a policy."""
    def __init__(self, validation, study, preference, cfg, folder, seed):
        super().__init__(verbose=0)
        if not validation or any(r.role != "validation" for r in validation):
            raise ValueError("Checkpoint selection accepts validation baskets only.")
        self.ranges, self.study, self.preference, self.cfg = validation, study, preference, cfg
        self.folder, self.seed = folder, int(seed)
        self.next_step = cfg.checkpoint_frequency
        self.best_score, self.best_step = -np.inf, None
        self.rows, self.basket_rows = [], []

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.evaluate_snapshot("initial", candidate=False)

    def _on_rollout_start(self):
        # This hook runs AFTER the preceding PPO update, not before that update.
        if self.num_timesteps >= self.next_step:
            self.evaluate_snapshot("periodic", candidate=True)
            while self.next_step <= self.num_timesteps:
                self.next_step += self.cfg.checkpoint_frequency

    def _on_training_end(self):
        self.evaluate_snapshot("final", candidate=True)
        if self.best_step is None:
            raise RuntimeError("No trained checkpoint was available.")
        (self.folder/"selection.json").write_text(json.dumps(dict(
            preference=self.preference.key, seed=self.seed, selected_step=self.best_step,
            final_step=int(self.num_timesteps), selected_validation_utility=self.best_score,
            final_validation_utility=self.final_score, initial_validation_utility=self.initial_score,
            selected_is_final=self.best_step == self.num_timesteps,
            selection="Mean realised preference utility across complete validation baskets",
            initial_eligible=False, test_used=False), indent=2))

    def evaluate_snapshot(self, stage, candidate):
        # Validation must not consume the random stream used by later training.
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        mode = self.model.policy.training
        try:
            measured = []
            common = dict(preference=self.preference.key, **asdict(self.preference), seed=self.seed,
                          step=int(self.num_timesteps), stage=stage, eligible=candidate)
            for r in self.ranges:
                values = metrics(run_policy(self.model, r, self.study.scaler, self.preference, self.cfg),
                                 self.preference, self.cfg)
                if not np.isfinite(values["preference_utility"]):
                    raise ValueError("Undefined validation utility. No basket was dropped.")
                measured.append(values)
                self.basket_rows.append(dict(common, basket_digest=r.panel.digest, **values))
            means = pd.DataFrame(measured).mean(numeric_only=True).to_dict()
            score = float(means["preference_utility"])
            improved = candidate and (self.best_step is None or score > self.best_score)
            if stage == "initial":
                self.initial_score = score
            if stage == "final":
                self.final_score = score
            if candidate and self.cfg.keep_checkpoints:
                (self.folder/"checkpoints").mkdir(exist_ok=True)
                self.model.save(self.folder/"checkpoints"/f"step_{self.num_timesteps:09d}.zip")
            if improved:
                self.model.save(self.folder/"selected.zip")
                self.best_score, self.best_step = score, int(self.num_timesteps)
            self.rows.append(dict(common, **means, selected_at_this_stage=bool(improved)))
            pd.DataFrame(self.rows).to_csv(self.folder/"validation_history.csv", index=False)
            pd.DataFrame(self.basket_rows).to_csv(self.folder/"validation_baskets.csv", index=False)
            print(f"  Validation {self.preference.key}, seed {self.seed}, step {self.num_timesteps:,}: "
                  f"utility {score:+.6f}"+(" [best]" if improved else ""))
        finally:
            self.model.policy.set_training_mode(mode)
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def library_defaults():
    """Capture installed-library defaults without manually recreating them."""
    defaults = {}
    for name, parameter in inspect.signature(PPO).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            value = parameter.default
            defaults[name] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    return defaults


def train(cfg, study):
    """Start a new run and fit each preference/seed independently."""
    cfg.validate()
    torch.set_num_threads(cfg.cpu_threads)
    training = [r for r in study.ranges if r.role == "training"]
    validation = [r for r in study.ranges if r.role == "validation"]
    run_folder = cfg.root/"reports"/datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run_folder.mkdir(parents=True, exist_ok=False)
    configuration = cfg.as_dict()
    for name in ("root", "show_figures"):
        configuration.pop(name)
    protocol = dict(configuration=configuration,
        files={role: [str(Path(p).relative_to(cfg.root)) for p in paths] for role, paths in study.files.items()},
        datasets=range_table(study, cfg).to_dict("records"),
        scaler=dict(feature_names=training[0].panel.feature_names,
                    mean=study.scaler.mean.tolist(), std=study.scaler.std.tolist()),
        source_hashes={p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        software={"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                  ["stable-baselines3", "gymnasium", "torch", "numpy", "pandas", "matplotlib", "reportlab", "wrds"]}},
        ppo_defaults=library_defaults(), ppo_overrides=cfg.ppo_overrides,
        reward=f"{cfg.reward_scale:g} * [net_return - A/2 * w'Cov_daily*w + B/{cfg.periods_per_year} * (mean(Q)-w'Q)]",
        action_mapping="amounts=(clip(action,-1,1)+1)/2; weights=amounts/sum(amounts); all-zero -> equal weight",
        covariance="Unadjusted historical sample covariance (ddof=1)")
    (run_folder/"protocol.json").write_text(json.dumps(protocol, indent=2))
    (run_folder/"library_defaults.json").write_text(json.dumps(protocol["ppo_defaults"], indent=2))
    range_table(study, cfg).to_csv(run_folder/"datasets.csv", index=False)
    rows = []
    for preference in cfg.preferences:
        for seed in cfg.model_seeds:
            folder = run_folder/"models"/preference.key/f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            env = TrainingEnv(training, study.scaler, preference, cfg)
            model = PPO(SharedStockPolicy, env, seed=int(seed), device=cfg.device,
                        policy_kwargs={"hidden_size": cfg.hidden_size}, **cfg.ppo_overrides)
            logger = configure(str(folder/"training_log"), ["csv"])
            model.set_logger(logger)
            try:
                model.save(folder/"initial.zip")
                callback = ValidationSelection(validation, study, preference, cfg, folder, seed)
                model.learn(total_timesteps=cfg.training_steps, callback=callback)
                model.save(folder/"final.zip")
                selection = json.loads((folder/"selection.json").read_text())
                row = dict(preference=preference.key, **asdict(preference), seed=int(seed),
                    actual_transitions=int(model.num_timesteps), seconds=time.perf_counter()-started,
                    selected_step=selection["selected_step"], selected_is_final=selection["selected_is_final"],
                    selected_validation_utility=selection["selected_validation_utility"],
                    final_validation_utility=selection["final_validation_utility"])
                # Record the actual resolved settings of this constructed PPO.
                actual = {name: getattr(model, name) for name in
                    ["n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef",
                     "max_grad_norm", "target_kl", "normalize_advantage", "learning_rate"]}
                actual["optimizer"] = type(model.policy.optimizer).__module__+"."+type(model.policy.optimizer).__name__
                actual["optimizer_defaults"] = model.policy.optimizer.defaults
                (folder/"resolved_ppo_settings.json").write_text(json.dumps(actual, indent=2, default=str))
                (folder/"fit.json").write_text(json.dumps(row, indent=2))
                rows.append(row)
                print(f"Completed {preference.key}, seed {seed}: {model.num_timesteps:,} transitions.")
            finally:
                logger.close()
                model.get_env().close()
                del model
            pd.DataFrame(rows).to_csv(run_folder/"training_record.csv", index=False)
    pd.DataFrame(rows).to_csv(run_folder/"training_record.csv", index=False)
    histories = [pd.read_csv(p) for p in sorted((run_folder/"models").glob("*/seed*/validation_history.csv"))]
    pd.concat(histories, ignore_index=True).to_csv(run_folder/"checkpoint_history.csv", index=False)
    print("Run folder:", run_folder)
    return run_folder
