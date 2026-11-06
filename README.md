# Financial and ESG PPO experiment

Open **MAIN_NOTEBOOK.ipynb** to configure and run the experiment. Section 1
contains the settings, followed by data collection, preparation, PPO training,
evaluation and report generation. Python calculations live in
`src/portfolio_experiment/`.

Existing data, models and reports remain in their current folders. Training
always starts a new run. The experiment uses separate, chronological training,
validation and testing periods.

## Notebook settings

| Section | Settings |
|---|---|
| 1.1 | Profile, PPO model seeds, training length, basket counts, checkpoint interval, risk aversion A and ESG appetite B |
| 1.2 | WRDS/saved data, stock seed, eligibility and score availability, table identifiers and chronological dates |
| 1.3 | Financial feature windows, estimation, trading rules, reward scale and utility solver convergence |
| 1.4 | PPO overrides, network width, device, threads and checkpoint storage |
| 1.5 | Construct, validate and display the full configuration |


| Profile | Model seeds | Requested transitions per fit | Training / validation / test baskets | Fits |
|---|---|---:|---|---:|
| Smoke | 7 | 2,048 | 1 / 1 / 1 | 2 |
| Pilot | 100, 101, 102 | 50,000 | 4 / 1 / 2 | 45 |
| Mass | 100–119 | 250,000 | 20 / 5 / 10 | 300 |

Mass and pilot use A = (1, 3, 6) and B = (0, 0.001, 0.005, 0.01, 0.1).
Smoke uses A = 3 and B = (0, 0.005).

`PPO_OVERRIDES = {}` uses the pinned library's PPO defaults. The notebook
displays those defaults and allows selected parameters to be overridden.

`RUN_TEST = False` omits final test-period evaluation. `KEEP_CHECKPOINTS`
controls storage of candidate snapshots, while initial, selected and final
policies are always saved.

## WRDS and saved baskets

WRDS collection requires an account with access to CRSP and Sustainalytics. The wrds Python package is included in requirements.txt and imported automatically by the collection module.

Set your username in the project’s .env file:

```text
WRDS_USERNAME=YOUR_WRDS_USERNAME
```

The password can be supplied through your existing .pgpass configuration or entered when WRDS prompts for it.

- `DATA_SOURCE = "wrds"` collects from WRDS. `REFRESH_WRDS_DATA = False` permits
  the local basket cache.
- `DATA_SOURCE = "saved"` loads the `SAVED_DATA_MANIFEST` supplied in Section 1.2.

Saved-panel loading lets new training runs use the same baskets without another
collection. The collector records selected seeds, rejected baskets, query/cache
status and data hashes. 

## Source modules

| Module | Responsibility |
|---|---|
| `config.py` | Experiment settings, preference grid and validation |
| `wrds_data.py` | Retrieval, cache, identifier links and score alignment |
| `data.py` | Historical features, sample covariance and training-fitted scaling |
| `finance.py` | Utility objective, numerical control optimisation and PPO reward |
| `environment.py` | Action-to-weight transformation, daily accounting and episode sampling |
| `policy.py` | Shared stock-processing neural network |
| `training.py` | New PPO fits, validation selection and saved policies |
| `evaluation.py` | Complete-period backtests, metrics and matched comparisons |
| `reporting.py` | Results tables, graph exports and PDF generation |
| `__init__.py` | Public workflow imports used by the notebook |

All modules are under `src/portfolio_experiment/`.

## Results and reporting

Collections are stored in `data/collections/<collection_id>/`. Each training
call creates a new `reports/<UTC_timestamp>/` folder containing:

| Output | Contents |
|---|---|
| `protocol.json`, `datasets.csv` | Settings, software/source versions, dataset identities and scaling |
| `models/` | Initial, selected and final policies, checkpoint scores and training records |
| `all_metrics.csv`, `summary.csv` | Individual evaluations and basket-balanced means |
| `paired_comparisons.csv`, `paired_summary.csv` | Matched PPO-minus-reference outcomes |
| `trajectories/`, `value_curves.csv` | Daily records and aggregated portfolio paths |
| `allocation_summary.csv`, `control_calculations.json` | Holding summaries and first-decision control inputs |
| `figures/`, `heatmap_values.csv`, `figure_scales.json`, `figure_manifest.json` | PNG/SVG/PDF figures, plotted values, colour limits and thesis figure mapping |
| `period_summary.csv`, `overall_financial_summary.csv`, `return_variation.csv` | Period comparisons, overall financial outcomes and basket variation |
| `MAIN_performance_report.pdf` | Parameters, results tables and figures |

