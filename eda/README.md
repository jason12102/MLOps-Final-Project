# Exploratory Data Analysis (EDA)

Exploratory analysis of the **raw** Pima Indians Diabetes dataset
(`data_versioning/outputs/diabetes_v1_raw.csv`). This is the *first* analytical step in the
project: it documents the data quality issues and statistical structure of the raw data, and
explains **why** the downstream `data_versioning` pipeline makes the choices it does
(median-imputing zeros in `clean.py`, binning BMI/Age in `features.py`).

## Why the raw version?
The raw v1 file keeps physiologically-impossible **zeros** (e.g. a BMI or BloodPressure of 0)
intact. These zeros are really *missing values* in disguise — analysing the raw data is what
reveals them. The cleaned/featured versions have already papered over this, so they hide the
very problem the EDA is meant to surface.

## Contents
`notebooks/diabetes_eda.ipynb` walks through 10 standard EDA tasks:

1. **Dataset overview** — shape, dtypes, sample rows, memory footprint.
2. **Summary statistics** — `describe()`; flags implausible minimums.
3. **Missing-value analysis** — counts/% of zero-as-missing per column.
4. **Target class balance** — `Outcome` distribution (~65% / 35%).
5. **Univariate distributions** — histograms + KDE for all 8 features.
6. **Outlier detection** — boxplots and IQR-based outlier counts.
7. **Correlation analysis** — Pearson heatmap; drivers of `Outcome`.
8. **Feature vs. target** — distributions split by diabetes outcome.
9. **Pairwise relationships** — pairplot of the top correlated features.
10. **Key insights** — findings tied back to the cleaning/feature pipeline.

## Running it

```bash
cd eda
pip install -r requirements.txt

# Ensure the raw data exists (regenerates it from data_versioning if needed):
python ../data_versioning/scripts/build_all.py   # only if outputs/diabetes_v1_raw.csv is missing

# Then launch the notebook:
jupyter lab notebooks/diabetes_eda.ipynb
# or run it headless:
jupyter nbconvert --to notebook --execute notebooks/diabetes_eda.ipynb
```

The notebook's first cell automatically locates `data_versioning/outputs/diabetes_v1_raw.csv`
and falls back to running the versioning pipeline if the file is absent.

## Data source
Pima Indians Diabetes Database (Kaggle): 768 records, 8 clinical features + binary `Outcome`.
