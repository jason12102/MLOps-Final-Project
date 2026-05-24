# Data Versioning — diabetes.csv

Three versions of the Pima Indians Diabetes dataset, each tracked by DVC and pinned to a git tag.

## Versions

| Git tag       | What it contains                                                                                   | Produced by                |
| ------------- | -------------------------------------------------------------------------------------------------- | -------------------------- |
| `v1-raw`      | Original Kaggle file, 768 × 9, `Outcome` target. Contains zero-as-missing in 5 medical columns.    | (none — untouched)         |
| `v2-cleaned`  | `v1` with zeros in `Glucose, BloodPressure, SkinThickness, Insulin, BMI` replaced by column median. | `scripts/clean.py`         |
| `v3-features` | `v2` + `BMI_category`, `Age_group`, `Glucose_BMI` (= Glucose × BMI).                               | `scripts/features.py`      |

The scripts overwrite `diabetes.csv` in place. Run them in sequence against the raw file to reproduce v2 and v3.

## Quickstart for teammates (all 3 versions in one command)

If you just want to inspect or compare v1, v2, and v3 side-by-side without juggling DVC tags:

```bash
pip install pandas numpy
# drop the raw Kaggle file at data_versioning/diabetes.csv
python data_versioning/scripts/build_all.py
```

This writes all three CSVs to `data_versioning/outputs/` (gitignored):

- `diabetes_v1_raw.csv`
- `diabetes_v2_cleaned.csv`
- `diabetes_v3_features.csv`

The per-step workflow below is still the canonical way to reproduce the DVC-tagged history.

## How a teammate reproduces all three versions on their laptop

The DVC remote in this repo is a folder on the original author's machine, so `dvc pull` won't work for you. Instead, regenerate the versions locally from the raw CSV — the scripts are deterministic, so you'll get byte-identical output.

```bash
# 1. clone and enter the repo
git clone <repo-url>
cd MLOps-Final-Project

# 2. install dependencies
pip install pandas numpy dvc

# 3. drop the original Kaggle file at data_versioning/diabetes.csv
#    (get it from Kaggle: Pima Indians Diabetes Database)
#    -> this is your v1-raw

# 4. produce v2-cleaned
python data_versioning/scripts/clean.py

# 5. produce v3-features
python data_versioning/scripts/features.py
```

After step 3 you have v1, after step 4 you have v2, after step 5 you have v3. To inspect any earlier version, restart from step 3 with a fresh copy of the raw CSV and stop at the appropriate step.

## How the version history is recorded in git

Each version is a git tag pointing to a commit that holds the matching `data_versioning/diabetes.csv.dvc` pointer (md5 + size of that version's CSV):

```bash
git fetch --tags
git log --oneline --decorate | grep -E "v1-raw|v2-cleaned|v3-features"
git show v2-cleaned:data_versioning/diabetes.csv.dvc   # see the recorded hash
```

If a shared DVC remote is configured in the future, `git checkout <tag> && dvc pull` will materialize the exact CSV for that tag.
