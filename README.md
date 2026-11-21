
# Satellite Re-entry Prediction

## Project Overview

### Introduction

Earth-orbiting satellites and other artificial objects gradually change their orbital characteristics due to several environmental and physical effects. One important phenomenon is **orbital decay**, where an object's orbit loses energy over time, potentially resulting in atmospheric re-entry.

Predicting when an object will decay and re-enter Earth's atmosphere is a challenging problem because orbital behaviour is influenced by multiple factors, including:

- Atmospheric drag.
- Orbital altitude.
- Orbital eccentricity.
- Orbital inclination.
- Solar and geomagnetic activity.
- Object characteristics, including size and mass.
- Changes in orbital parameters over time.

This project aims to investigate satellite orbital behaviour using historical Two-Line Element (TLE) and General Perturbations (GP/OMM) data, perform data analysis and feature engineering, and eventually develop a machine learning model capable of estimating satellite decay or re-entry timelines.

The central idea is to use historical orbital observations to understand how orbital parameters evolve and identify patterns associated with orbital decay.

### Project Objectives

1. Collect historical orbital data for satellites and other catalogued space objects.
2. Build a structured dataset from historical TLE/OMM records.
3. Integrate satellite decay information from an external catalogue.
4. Validate and investigate the quality of the collected data.
5. Perform exploratory data analysis (EDA).
6. Identify useful orbital parameters and derive additional features.
7. Investigate relationships between orbital characteristics and decay behaviour.
8. Develop and evaluate a predictive model for satellite decay.

---

## Data Sources

### 1. Space-Track

Space-Track provides historical orbital data through its API.

The project uses historical General Perturbations data, including orbital elements and associated satellite metadata.

Important fields include:

- `NORAD_CAT_ID`
- `OBJECT_NAME`
- `OBJECT_ID`
- `EPOCH`
- `MEAN_MOTION`
- `ECCENTRICITY`
- `INCLINATION`
- `RA_OF_ASC_NODE`
- `ARG_OF_PERICENTER`
- `MEAN_ANOMALY`
- `BSTAR`
- `MEAN_MOTION_DOT`
- `MEAN_MOTION_DDOT`

The orbital records provide observations of an object's orbital state at different epochs.

### 2. CelesTrak

CelesTrak provides a catalogue containing satellite information, including decay dates for catalogued objects.

The catalogue is used to supplement the Space-Track dataset with decay information.

The mapping is performed using:

| Space-Track      | CelesTrak                |
| ---------------- | ------------------------ |
| `NORAD_CAT_ID` | `NORAD Catalog Number` |
| —               | `Decay Date`           |

The catalogue decay date is stored separately as `CATALOG_DECAY_DATE`.

The original `DECAY_DATE` field from Space-Track is preserved without modification.

---

## Repository Structure

```text
satelite/
│
├── data/
│   ├── raw/
│   │   ├── tle_history/
│   │   │   └── Historical satellite TLE/OMM JSON files
│   │   │
│   │   └── decayed_satellites.csv
│   │
│   ├── preprocessed/
│   │   ├── tle_features.csv
│   │   └── tle_features.parquet
│   │
│   └── curated/
│       ├── tle_features.csv
│       └── tle_features.parquet
│
├── src/
│   └── data_processing/
│       ├── TLE_json_to_parquet.py
│       ├── 02_database_validation.py
│       └── 03_data_profiling.py
│
├── reports/
│   └── eda/
│       └── plots/
│
└── README.md
```

### Directory Responsibilities

| Directory                 | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `data/raw/`             | Original downloaded data                           |
| `data/raw/tle_history/` | Historical orbital records in JSON format          |
| `data/preprocessed/`    | Merged and feature-engineered dataset              |
| `data/curated/`         | Working dataset used for analysis                  |
| `src/data_processing/`  | Data processing, validation, and profiling scripts |
| `reports/eda/`          | Exploratory data analysis reports                  |
| `reports/eda/plots/`    | Generated visualizations                           |

**Data preservation principle:** Raw data is retained unchanged. Analysis and future feature-selection operations should produce separate outputs rather than silently modifying the original dataset.

---

# Data Processing Pipeline

## 1. Historical Data Collection

Historical TLE/OMM records are collected and stored as JSON files under:

```text
data/raw/tle_history/
```

The collection contains historical observations for multiple satellites.

Each original orbital record is treated as an individual observation.

No records are intentionally deduplicated during dataset construction.

## 2. JSON Processing and Dataset Construction

The JSON files are read and converted into a structured tabular dataset.

The processing pipeline:

- Iterates through the historical JSON files.
- Extracts orbital records.
- Preserves original API fields and values.
- Combines observations into a single dataset.
- Associates records with their corresponding satellite identifiers.

The resulting dataset is stored in CSV and Parquet formats.

## 3. Feature Engineering

Additional orbital parameters are calculated from the available orbital elements.

The following calculations are implemented:

### Semi-major axis

Using mean motion:

\[
n = \text{MEAN\_MOTION}\times\frac{2\pi}{86400}
\]

\[
a = \left(\frac{\mu}{n^2}\right)^{1/3}
\]

Where:

- \(a\) = semi-major axis in kilometres.
- \(n\) = mean motion in radians per second.
- \(\mu\) = Earth's gravitational parameter.

\[
\mu = 398600.4418\ km^3/s^2
\]

### Perigee altitude

\[
h_p = a(1-e)-R_E
\]

### Apogee altitude

\[
h_a = a(1+e)-R_E
\]

Where:

- \(e\) = orbital eccentricity.
- \(R_E = 6378.137\ km\) = Earth's equatorial radius.

These derived values provide additional information about the orbital geometry.

## 4. Decay Date Integration

The CelesTrak decay catalogue is merged with the historical orbital dataset using the NORAD catalogue identifier.

The resulting field is:

```text
CATALOG_DECAY_DATE
```

The original Space-Track `DECAY_DATE` field is retained separately.

A derived field, `DAYS_TO_DECAY`, was also calculated using the difference between the catalogue decay date and orbital epoch:

\[
DAYS\_TO\_DECAY =
\frac{t_{\text{decay}}-t_{\text{epoch}}}{86400}
\]

This represents the time between an orbital observation and the recorded decay date.

It is a derived target-related quantity, not an independent orbital measurement.

## 5. Dataset Formats

Two formats are maintained:

| Format  | Intended use                              |
| ------- | ----------------------------------------- |
| CSV     | Human-readable inspection and portability |
| Parquet | Efficient analytical processing           |

Parquet is the primary format for EDA and subsequent machine learning workflows.

---

# Data Validation

The dataset was validated before exploratory analysis.

Validation covered:

- Expected column presence.
- Row and column counts.
- Missing values.
- Numeric and date validity.
- Orbital parameter ranges.
- TLE line checksums.
- Duplicate satellite and epoch combinations.
- CSV and Parquet consistency.

### Validation Results

| Metric                          |  Result |
| ------------------------------- | ------: |
| Total records                   | 950,677 |
| Total columns                   |      41 |
| Unique satellites               |     204 |
| Invalid TLE checksums           |       0 |
| Invalid numeric values detected |       0 |
| CSV/Parquet row-count mismatch  |    None |
| CSV/Parquet column mismatch     |    None |

Duplicate `(NORAD_CAT_ID, EPOCH)` combinations were observed.

These were flagged for investigation but not automatically removed. Identical epochs do not, by themselves, establish that two observations are redundant or invalid.

### Missing Values

| Column           | Missing records |
| ---------------- | --------------: |
| `RCS_SIZE`     |           9,600 |
| `COUNTRY_CODE` |           1,167 |
| `LAUNCH_DATE`  |           1,186 |
| `SITE`         |           1,167 |
| `DECAY_DATE`   |         950,476 |

The high missing percentage in the original `DECAY_DATE` field is expected from the source data.

The separate `CATALOG_DECAY_DATE` field was successfully mapped for all 950,677 records.

---

# Exploratory Data Analysis (EDA)

## Current Status

The initial EDA stage has been completed.

The profiling script is:

```text
src/data_processing/03_data_profiling.py
```

It reads:

```text
data/curated/tle_features.parquet
```

Generated reports and visualizations are saved under:

```text
reports/eda/
```

The EDA process does not modify the curated dataset.

### Analysis Performed

- Dataset overview.
- Column profiling.
- Missing-value analysis.
- Data type inspection.
- Numeric conversion.
- Orbital parameter statistics.
- Constant-column identification.
- Numerical distribution plots.
- Epoch coverage analysis.

A separate `analysis_df` copy is used for numerical and datetime analysis. This preserves the original dataset's stored values and data types.

## Dataset Overview

| Metric              |     Value |
| ------------------- | --------: |
| Rows                |   950,677 |
| Columns             |        41 |
| Memory usage        | 716.79 MB |
| Unique NORAD IDs    |       204 |
| Unique epochs       |   773,502 |
| Unique object names |       197 |
| Unique object IDs   |       214 |

### Epoch Coverage

| Metric         | Result     |
| -------------- | ---------- |
| Earliest epoch | 1969-10-15 |
| Latest epoch   | 2026-11-18 |
| Valid epochs   | 950,677    |
| Invalid epochs | 0          |

The dataset covers a substantial historical period, although the distribution of observations across years is not necessarily uniform.

A yearly analysis is still required to understand the temporal distribution of the collected records.

## Orbital Parameter Statistics

The following statistics were obtained after converting orbital parameters into numeric types.

| Parameter              |         Mean |       Median |      Minimum |       Maximum |
| ---------------------- | -----------: | -----------: | -----------: | ------------: |
| Mean Motion            |       14.881 |       15.064 |        2.010 |        16.556 |
| Eccentricity           |       0.0136 |     0.000488 |     0.000001 |      0.753049 |
| Inclination            |     70.715° |     69.923° |      3.204° |      99.102° |
| RA of Ascending Node   |    180.220° |    179.956° |            0 |      359.9996 |
| Argument of Pericenter |    147.338° |    105.802° |       0.0001 |      359.9995 |
| Mean Anomaly           |    207.871° |    248.090° |       0.0003 |      359.9990 |
| BSTAR                  |     0.000959 |     0.000383 |    -0.418040 |      1.202068 |
| Semi-major Axis        | 7,100.406 km | 6,925.345 km | 6,502.793 km | 26,520.611 km |
| Period                 |  100.442 min |   95.592 min |   86.978 min |   716.366 min |
| Apogee                 |   923.994 km |   548.201 km |   130.620 km | 40,113.803 km |
| Perigee                |   520.548 km |   545.719 km |   104.696 km |  1,937.542 km |

### Initial Observations

**1. Eccentricity**

The median eccentricity is approximately 0.000488.

Most observations therefore correspond to nearly circular orbits. However, the maximum eccentricity of 0.753049 indicates that some objects have substantially more elliptical orbits.

**2. Orbital altitude**

The median perigee altitude is approximately 545.7 km, while the median apogee altitude is approximately 548.2 km.

This indicates that the typical observation in this dataset is from a relatively low-eccentricity orbit.

**3. Orbital period**

The median orbital period is approximately 95.6 minutes.

**4. BSTAR**

BSTAR has a wide range, including negative and unusually large values.

Its distribution and extreme observations require further investigation before using it as a predictive feature.

**5. Temporal coverage**

Although the dataset extends back to 1969, observation density may be concentrated in recent years. This needs to be quantified rather than inferred from the date range alone.

## Constant Columns

The following columns have only one unique value in the current dataset:

```text
CCSDS_OMM_VERS
COMMENT
ORIGINATOR
CENTER_NAME
REF_FRAME
TIME_SYSTEM
MEAN_ELEMENT_THEORY
EPHEMERIS_TYPE
CLASSIFICATION_TYPE
ELEMENT_SET_NO
```

These fields currently provide no variation for a predictive model.

They are retained in the curated dataset for source completeness. Their exclusion from a future model feature matrix can be considered separately.

---

# Current Project Progress

<details>
<summary>Completed work</summary>

- [X] Historical orbital data collection.
- [X] Storage of historical records as JSON.
- [X] JSON processing and tabular dataset construction.
- [X] Preservation of original API fields.
- [X] Semi-major axis calculation.
- [X] Perigee altitude calculation.
- [X] Apogee altitude calculation.
- [X] CelesTrak decay catalogue integration.
- [X] Separate `CATALOG_DECAY_DATE` field.
- [X] CSV and Parquet dataset generation.
- [X] Dataset validation.
- [X] TLE checksum validation.
- [X] CSV and Parquet consistency checks.
- [X] Curated dataset preparation.
- [X] Initial data profiling.
- [X] Numeric conversion checks.
- [X] Orbital parameter descriptive statistics.
- [X] Constant-column identification.
- [X] Initial epoch coverage analysis.

</details>

<details>
<summary>Remaining work</summary>

- [ ] Inspect numerical distribution plots.
- [ ] Investigate extreme orbital values.
- [ ] Examine BSTAR distribution and outliers.
- [ ] Analyse observation counts per year.
- [ ] Analyse unique satellite counts per year.
- [ ] Investigate individual satellite orbital histories.
- [ ] Perform correlation analysis.
- [ ] Study orbital decay trends.
- [ ] Identify useful predictive features.
- [ ] Prepare a separate model-ready dataset.
- [ ] Establish appropriate training and testing strategies.
- [ ] Develop baseline predictive models.
- [ ] Evaluate model performance.
- [ ] Investigate prediction accuracy over different time horizons.

</details>

---

# Next Steps

## 1. Complete EDA

The next stage is to investigate the distributions and relationships between orbital parameters.

Particular attention should be given to:

- Eccentricity.
- BSTAR.
- Mean motion.
- Semi-major axis.
- Perigee altitude.
- Apogee altitude.
- Mean motion derivatives.

Extreme values should be investigated before deciding whether they represent valid orbital behaviour, unusual observations, or data-quality problems.

## 2. Temporal Analysis

A major part of this project is understanding how orbital parameters evolve over time.

The next analysis should include:

- Number of observations per year.
- Number of unique satellites per year.
- Observation frequency for individual satellites.
- Changes in perigee altitude over time.
- Changes in mean motion over time.
- Changes in BSTAR over time.

Individual satellite histories are especially important because orbital decay is a temporal process.

## 3. Correlation and Feature Analysis

Investigate relationships between orbital parameters and decay-related quantities.

A correlation matrix across the entire dataset can be useful for initial exploration, but it may mix differences between satellites with changes occurring within individual satellites.

Both global and per-satellite analysis should therefore be considered.

## 4. Feature Selection and Target Definition

Before training a model, the prediction task must be defined precisely.

For example:

- Predicting the remaining time until decay.
- Predicting a future decay date.
- Predicting whether an object will decay within a specified period.

`CATALOG_DECAY_DATE` and `DAYS_TO_DECAY` are target-related fields. They must not be included as ordinary input features when predicting decay time.

The model should only use information available at the prediction epoch.

## 5. Model Development

After EDA and feature selection:

1. Create a separate model-ready dataset.
2. Establish a baseline model.
3. Choose suitable temporal and satellite-aware validation strategies.
4. Train candidate models.
5. Evaluate prediction errors.
6. Compare performance across satellites and orbital regimes.

Care is required when splitting the data. Randomly splitting individual TLE records can put observations from the same satellite and similar epochs into both training and testing sets, producing overly optimistic results.

---

# Technical Environment

The project uses Python and a data-processing workflow based on Pandas and Parquet.

The scripts are executed using `uv`:

```bash
uv run src/data_processing/TLE_json_to_parquet.py
uv run src/data_processing/02_database_validation.py
uv run src/data_processing/03_data_profiling.py
```

The current workflow uses:

- Python
- Pandas
- NumPy
- PyArrow
- Matplotlib
- Parquet
- JSON
- CSV

---

# Important Data Handling Decisions

1. Original raw JSON files are preserved.
2. Original API fields are not overwritten during feature engineering.
3. Original `DECAY_DATE` and catalogue-derived `CATALOG_DECAY_DATE` remain separate.
4. Duplicate satellite/epoch combinations are not automatically removed.
5. The curated dataset is not modified by EDA scripts.
6. Parquet is the primary analytical format.
7. CSV is retained for inspection and portability.
8. Feature selection will produce a separate model-ready dataset.
9. Target-related fields must be excluded from model inputs to prevent data leakage.

---

# Current Project Status

**Stage: Data collection, preprocessing, validation, and initial EDA completed.**

The project currently has a structured historical orbital dataset containing approximately 950,000 observations across 204 satellites.

Historical orbital information has been combined with catalogue decay dates, validated, and prepared for exploratory analysis.

The next major milestone is to understand the temporal behaviour of individual satellites and determine which orbital parameters provide useful information about future orbital decay.

The machine learning stage has not yet started.

---

*This README reflects the current development state of the project. Update the progress checklist and findings as subsequent analysis and modelling stages are completed.*
