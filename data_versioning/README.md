# Data Versioning — diabetes.csv

Three DVC-tracked snapshots of the Pima Indians Diabetes dataset, each pinned to a git tag. The actual CSV is stored in the Google Drive DVC remote; git only tracks the `.dvc` pointer.

## Versions

| Git tag       | What it contains                                                                                   | Produced by                |
| ------------- | -------------------------------------------------------------------------------------------------- | -------------------------- |
| `v1-raw`      | Original Kaggle file, 768 × 9, `Outcome` target. Contains zero-as-missing in 5 medical columns.    | (none — untouched)         |
| `v2-cleaned`  | `v1` with zeros in `Glucose, BloodPressure, SkinThickness, Insulin, BMI` replaced by column median. | `scripts/clean.py`         |
| `v3-features` | `v2` + `BMI_category`, `Age_group`, `Glucose_BMI` (= Glucose × BMI).                               | `scripts/features.py`      |

The scripts overwrite `diabetes.csv` in place — they're meant to be run in
sequence against a DVC checkout, not as a standalone pipeline.

## Switching between versions (teammates)

```bash
git fetch --tags
git checkout v2-cleaned        # or v1-raw / v3-features
dvc pull                       # downloads the matching CSV from Drive
```

Once a version's data is in the local DVC cache, `dvc checkout` is enough on
subsequent switches (no network).

## Access required

To use this module you need **both**:

1. **GitHub collaborator** access to this repo — for the code, the `.dvc`
   pointer files, and the version tags.
2. **Google Drive "Editor"** access to the DVC remote folder — for the actual
   data blobs. Without this, `dvc pull` will fail with a permission error.

## Colab usage

```bash
!pip install -q "dvc[gdrive]"
!git clone <repo-url> && cd MLOps-Final-Project && git checkout v2-cleaned
!dvc pull   # first run opens a Google OAuth flow — use the account with Editor access
```
