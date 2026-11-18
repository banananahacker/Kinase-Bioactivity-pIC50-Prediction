# Kinase Bioactivity ($pIC_{50}$) Prediction Pipeline (v2)

An end-to-end machine learning pipeline for retrieving, curating, standardizing, and modeling small-molecule bioactivity data from the ChEMBL database. This project predicts half-maximal inhibitory concentration ($pIC_{50}$) values across three clinically relevant receptor tyrosine kinases: **EGFR** (`CHEMBL203`), **VEGFR2** (`CHEMBL279`), and **HER2** (`CHEMBL1824`).

To prevent data leakage and accurately simulate real-world drug discovery generalization across novel chemical series, the pipeline uses **Murcko Scaffold Splitting** and **Group K-Fold Cross-Validation**.

---

## Project Structure

* `pull_chembl_raw.py` — Queries the ChEMBL API for uncensored (`=`), nanomolar (`nM`) $IC_{50}$ records for a given target ID and exports them to `<TARGET>_raw.csv`.
* `filter_chembl_clean.py` — Multi-core data curation and chemical standardization script using RDKit. Removes mutant assays, non-binding assays, duplicates, inorganic/acyclic compounds, and extreme statistical outliers, outputting `<TARGET>_clean.csv`.
* `train_model_v2.py` — Generates Morgan fingerprints and chemical descriptors, performs an 80/20 Murcko scaffold split, benchmarks multiple regression models via 5-fold `GroupKFold` cross-validation, and evaluates the top model against a mean-baseline on the held-out test set.
* `predict.py` — Trains a Random Forest on `<TARGET>_clean.csv` (same features and settings as `train_model_v2.py`) and predicts the $pIC_{50}$ of a single molecule from its SMILES string.

---

## Methodology & Pipeline Architecture

### 1. Data Curation & Standardization (`filter_chembl_clean.py`)
Raw ChEMBL bioactivity data contains significant experimental noise, varying assay protocols, and structural inconsistencies. The curation pipeline enforces strict filtering criteria:
* **Wild-Type Binding Consistency:** Excludes any record where `assay_variant_mutation` is populated or where `assay_description` references mutations (`mutant`, `L858R`, `T790M`, `L861Q`, `C797S`). Restricts data strictly to Assay Type `B` (Binding) to eliminate ~200x potency discrepancies observed between functional (`F`) and binding (`B`) assays.
* **Deduplication:** Filters out ChEMBL-flagged `potential_duplicate` records.
* **Chemical Standardization:** Parallelized across CPU cores using RDKit's `rdMolStandardize`. Strips salts (`LargestFragmentChooser`), neutralizes charges (`Uncharger`), and filters out inorganic molecules (must contain carbon) and acyclic molecules (must contain at least one ring to produce a valid Murcko scaffold).
* **Target Transformation & Outlier Removal:** Converts $IC_{50}$ (nM) to $pIC_{50}$ via $pIC_{50} = 9 - \log_{10}(IC_{50\text{ [nM]}})$. Removes extreme values ($3 < pIC_{50} < 12$”).
* **Aggregation:** Groups duplicate standardized SMILES strings and takes the **median** $pIC_{50}$ (alongside standard deviation and measurement count) to ensure robustness against repeated measurements.

#### Data Funnel Across Targets
| Curation Step | EGFR (`CHEMBL203`) | VEGFR2 (`CHEMBL279`) | HER2 (`CHEMBL1824`) |
| :--- | :---: | :---: | :---: |
| **Raw Activity Records** | 19,647 | 13,183 | 3,385 |
| No Mutants in Description | 13,533 | 13,171 | 3,334 |
| Assay Type `B` Only | 12,268 | 12,790 | 3,064 |
| No Potential Duplicates | 10,966 | 11,475 | 2,737 |
| No Assay Variant Mutations | 10,934 | 11,475 | 2,737 |
| Valid Standardized SMILES | 10,917 | 11,471 | 2,730 |
| Contains Carbon Atoms | 10,916 | 11,471 | 2,730 |
| $3 < pIC_{50} < 12$ | 10,869 | 11,468 | 2,730 |
| Contains $\ge 1$ Ring (Valid Scaffold) | 10,868 | 11,468 | 2,730 |
| **Final Unique Molecules (Clean)** | **8,628** | **9,569** | **2,313** |

---

### 2. Featurization & Scaffold Splitting (`train_model_v2.py`)
* **Molecular Features (2,051 dimensions):**
  * **2,048-bit Morgan Fingerprints** (Radius = 2).
  * **3 Physicochemical Descriptors:** Molecular Weight (`MolWt`), Octanol-Water Partition Coefficient (`MolLogP`), and Topological Polar Surface Area (`TPSA`).
* **Murcko Scaffold Split (80% Train / 20% Test):**
  * Molecules are grouped by their core Murcko scaffold (`MurckoScaffoldSmiles`).
  * Scaffold groups are sorted by size in descending order and allocated to the training set until the 80% threshold is reached; all remaining (rarer/novel) scaffolds are placed in the 20% test set. This guarantees that no core chemical scaffold is shared between training and testing and prevents testing data leaks.

---

### 3. Model Selection & Parameter Tuning
All models are evaluated on the training split using 5-fold **GroupKFold Cross-Validation** grouped by Murcko scaffold:
1. **Baseline Model:** Predicts the mean training $pIC_{50}$ for all inputs to establish a zero-correlation floor ($R^2 \approx 0$).
2. **Random Forest Regressor:** `n_estimators=500, max_depth=None, min_samples_leaf=1`. Tuning showed unconstrained trees outperformed depth-limited (`max_depth=10, 15`) and leaf-pruned (`min_samples_leaf=3, 8`) variants.
3. **Ridge Regression Pipeline:** Uses a `ColumnTransformer` to apply `StandardScaler` exclusively to the 3 continuous descriptors while passing the 2,048 binary fingerprint bits through unmodified. Tuned across $\alpha \in \{0.1, 10, 100, 1000\}$, with $\alpha = 100$ achieving optimal regularization.
4. **Histogram-Based Gradient Boosting (`HistGradientBoostingRegressor`):** Tuned across different `learning_rate`, `max_iter`, and `max_leaf_nodes` and showed that `learning_rate=0.05, max_iter=600, max_leaf_nodes=50` outperformed other variants.

Notes: All hyperparameter tuning was done on EGFR data and reused unchanged for VEGFR2 and HER2.

---

# 4. Predict pIC50 for a new molecule (target first, then a quoted SMILES)
python predict.py CHEMBL203 "COc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OC"

Notes: `predict.py` retrains the model on each call, using the 80% scaffold-split training set. Predictions for molecules unlike the training data (novel scaffolds) are less reliable, with test $R^2$ around 0.45-0.54 (see Results). This is such that model is never evaluated on testing set to prevent data leaks.

## Results & Benchmarks

### Cross-Validation Performance (5-Fold `GroupKFold` $R^2$)
| Target | Train / Test Split | Ridge ($\alpha=100$) | HistGradientBoosting | Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **EGFR** (`CHEMBL203`) | 6,902 / 1,726 | $0.5084 \pm 0.0579$ | **$0.6068 \pm 0.0441$** | $0.5990 \pm 0.0542$ |
| **VEGFR2** (`CHEMBL279`) | 7,655 / 1,914 | $0.4055 \pm 0.0420$ | $0.4828 \pm 0.0386$ | **$0.4971 \pm 0.0346$** |
| **HER2** (`CHEMBL1824`) | 1,850 / 463 | $0.5562 \pm 0.0088$ | $0.5652 \pm 0.0114$ | **$0.5773 \pm 0.0313$** |

### Held-Out Scaffold Test Set Performance (Random Forest)
| Target | Train RMSE | Train $R^2$ | Baseline Test RMSE | Baseline Test $R^2$ | **Test RMSE** | **Test $R^2$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EGFR** (`CHEMBL203`) | 0.2630 | 0.9640 | 1.3292 | -0.0297 | **0.9519** | **0.4720** |
| **VEGFR2** (`CHEMBL279`) | 0.2305 | 0.9586 | 1.1645 | -0.0186 | **0.8589** | **0.4459** |
| **HER2** (`CHEMBL1824`) | 0.2359 | 0.9587 | 1.2420 | -0.0005 | **0.8441** | **0.5378** |

### Key Takeaways
* **Generalization to Novel Scaffolds:** Across all three targets, the models substantially outperform the mean baseline, achieving test $R^2$ values between **0.45 and 0.54** and RMSEs of **0.84 to 0.95** on completely unseen chemical scaffolds in the isolated testing group.
* **Random Forest vs. Gradient Boosting on Rare Scaffolds:** On EGFR, `HistGradientBoostingRegressor` achieved a slightly higher cross-validation score ($R^2 = 0.6068$) than `RandomForestRegressor` ($R^2 = 0.5990$). However, when evaluated on the held-out test set—which is populated by the rarest and smallest scaffold clusters due to size-sorted scaffold splitting—Random Forest generalized better (Test $R^2 = 0.4720$, $\text{RMSE} = 0.9519$) than Gradient Boosting (Test $R^2 = 0.4404$, $\text{RMSE} = 0.9799$). Random Forest also led in cross-validation for both VEGFR2 and HER2.

## Limitations
- Hyperparameters were tuned on EGFR only and reused for VEGFR2 and HER2.
- Possible overfitting to training data

---

## Installation & Usage

### Prerequisites
Python 3.11. Install dependencies:
```bash
pip install -r requirements.txt
```
If the `rdkit` pip install fails, use conda instead: `conda install -c conda-forge rdkit`.

### Running the Pipeline
Pass any valid ChEMBL Target ID as a command-line argument (defaults to `CHEMBL203` if omitted):

```bash
# 1. Pull raw IC50 data from ChEMBL
python pull_chembl_raw.py CHEMBL203

# 2. Filter, standardize SMILES, and aggregate duplicates
python filter_chembl_clean.py CHEMBL203

# 3. Perform scaffold split, cross-validation, and test evaluation
python train_model_v2.py CHEMBL203
```

To run the pipeline for **VEGFR2** or **HER2**, replace `CHEMBL203` with `CHEMBL279` or `CHEMBL1824` respectively.

---

## Future Improvements
* **Unit Conversion in Raw Data Pull:** Expand `pull_chembl_raw.py` to pull all convertible molar concentration units (e.g., $\mu\text{M}$, $\text{pM}$, $\text{M}$) and standardize to $\text{nM}$ rather than filtering strictly for `standard_units="nM"`.
* **Tautomer Canonicalization:** Evaluate enabling RDKit's `TautomerEnumerator().Canonicalize()` during SMILES cleanup (currently disabled for computational speed) to merge tautomeric duplicates.
