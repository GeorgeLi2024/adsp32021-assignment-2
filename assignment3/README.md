# ADSP 32021 Assignment 3: AutoML

This folder contains the A3 report source/PDF and the working Python scripts. The selected workflow is full-code Python: FLAML AutoML, with Feast for feature storage and MLflow for tracking, plus the required H2O AutoML run. The report states which values came from the retained October 18, 2026 run and which candidate-level FLAML speed measurements were made on November 3.

## Data source

The course dataset is `athletes.csv.zip`; retained archive SHA-256: `4a7a2701564ca0c7d2bc5016964aab6997abacf25ac74abaa28deda8c6689dd6`. The archive member `athletes.csv` SHA-256 is `e6a019bd1d2b6c29bf02a62693d22443850b12a1c475be9f68c34892bfc6ff83`. The dataset is not included in this code folder.

The A3 AutoML prompt does not specify a target formula; for comparison with Part A, this package retains `total_lift = candj + snatch + deadlift + backsq`. The source data provides no unit metadata. The input filters, two feature versions, and gender-stratified split are implemented in `code/prepare_data.py`.

## Reproduce the AutoML runs

Use Python 3.12 for the main pipeline, Python 3.11 for H2O, and Java 17. Create isolated environments:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements-main.txt
python3.11 -m venv .venv-h2o
.venv-h2o/bin/python -m pip install -r code/requirements-h2o.txt
```

Then run from this folder with a course-supplied input archive and a new output directory:

```sh
.venv/bin/python code/run_assignment.py \
  --input /path/to/athletes.csv.zip \
  --output /path/to/new-output \
  --h2o-python /path/to/.venv-h2o/bin/python \
  --java-home /path/to/java-17 \
  --ml-time-budget 120 \
  --ml-max-iter 30 \
  --h2o-time-budget 120 \
  --h2o-max-models 6
```

For a completed run directory, `code/refit_flaml_trial_speed.py --run-dir /path/to/new-output --output-dir /path/to/new-speed-output --n-jobs 1` reads the logged FLAML trial configurations, refits each on the training split, and benchmarks the shared 1,000-row validation batch. It refuses to overwrite its output directory. The trial-level validation scores remain those recorded during the original AutoML search; speed is measured after refit.

## Retained-run provenance

The report uses a completed October 18, 2026 run for the AutoML model and H2O leaderboard results. FLAML's original trial logs preserve the validation ranking for every configuration. The November 3 refit benchmark measured all 60 distinct configurations and confirmed all 60 validation scores matched the original logs. The report distinguishes these measurements from a new AutoML search.
