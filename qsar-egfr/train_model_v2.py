from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "CHEMBL203"  
data = pd.read_csv(f"{TARGET}_clean.csv")
print(f"Training data for {TARGET}")
size_fp = 2048
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size_fp) #sets up fingerprinting

skipped_smiles = 0

data_simple = zip(data["smiles"],data["pIC50"]) #only gets smiles and pic50s

x_list = [] #list of fingerprints
y_list = [] #list of pic50s
smi_list = [] #list of smiles
scaffold_list = [] #list of scaffoled smiles

for smi, pIC in data_simple:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
            skipped_smiles += 1
            continue
    mol_fped = fp_gen.GetFingerprintAsNumPy(mol)
    mol_descript = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])
    mol_fped = np.concatenate((mol_fped, mol_descript))
    x_list.append(mol_fped)
    y_list.append(pIC)
    smi_list.append(smi)
    scaffold_list.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))


x = np.vstack(x_list)
y = np.array(y_list)
smiles = np.array(smi_list)
scaffolds = pd.Series(scaffold_list)

dict_scaffolds = {}

for i,s in enumerate(scaffolds):
    if(s not in dict_scaffolds):
         dict_scaffolds[s] = []
    dict_scaffolds[s].append(i)

sorted_dict_scaffolds = sorted(dict_scaffolds.values(),key=len,reverse=True)

train_idx = []
test_idx = []
target = 0.8 * len(y)
train_scaffolds = []

for group in sorted_dict_scaffolds:
    if (len(group) + len(train_idx) <= target):
          train_idx.extend(group)
          train_scaffolds.extend(scaffolds[group])
    else:
         test_idx.extend(group)

print(f"training data length: {len(train_idx)}")
print(f"testing data length {len(test_idx)}\n")

rf = RandomForestRegressor(
     n_estimators=500,
     random_state=42,
     n_jobs=10,
     max_depth=None
)

rf.fit(x[train_idx], y[train_idx])

train_prediction = rf.predict(x[train_idx])
train_rmse = np.sqrt(mean_squared_error(y[train_idx],train_prediction))
train_rSquared = r2_score(y[train_idx],train_prediction)
'''
test_prediction = rf.predict(x[test_idx])
test_rmse = np.sqrt(mean_squared_error(y[test_idx],test_prediction))
test_rSquared = r2_score(y[test_idx], test_prediction)
'''
print(f"train RMSE: {train_rmse:.4f} | train R²: {train_rSquared:.4f} \n")

#print(f"test RMSE: {test_rmse:.4f} | test R²: {test_rSquared:.4f}")

baseline = np.full(len(y[train_idx]), np.mean(y[train_idx]))
rf_stupid = RandomForestRegressor(
     n_estimators=500,
     random_state=42,
     n_jobs=10,
     max_depth=None
)
rf_stupid.fit(x[train_idx], baseline)
stupid_prediction = rf_stupid.predict(x[test_idx])
stupid_rmse = np.sqrt(mean_squared_error(y[test_idx], stupid_prediction))
stupid_rSquared = r2_score(y[test_idx], stupid_prediction)
print(f"baseline RMSE: {stupid_rmse:.4f} | baseline R²: {stupid_rSquared:.4f}")

gkf = GroupKFold(n_splits = 5)
cvs = cross_val_score(rf, x[train_idx], y[train_idx], cv = gkf, groups = train_scaffolds, scoring = "r2")
print(f"random forest regression cross val score std: {np.std(cvs):.4f}")
print(f"random forest regression cross val score mean: {np.mean(cvs):.4f} \n")

cols = [size_fp, size_fp+1, size_fp+2]
transformer = ColumnTransformer(
     transformers=[
          ('num', StandardScaler(), cols)
     ],
     remainder='passthrough'
)

pipeline = Pipeline(steps=[
     ('preprocessor', transformer),
     ('regressor', Ridge(alpha=100))
])

rd_cvs = cross_val_score(pipeline, x[train_idx], y[train_idx], cv = gkf, groups = train_scaffolds, scoring="r2")
print(f"ridge pipeline cross val score std: {np.std(rd_cvs):.4f}")
print(f"ridge pipeline cross val score mean: {np.mean(rd_cvs):.4f}\n")

gb = HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=False,
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=50
    )

gb_cvs = cross_val_score(gb, x[train_idx], y[train_idx], cv=gkf, groups=train_scaffolds, scoring="r2")
print(f"gradient boosting cross val score std: {np.std(gb_cvs):.4f}")
print(f"gradient boosting cross val score mean: {np.mean(gb_cvs):.4f}\n")


rf.fit(x[train_idx], y[train_idx])
prediction = rf.predict(x[test_idx])
rmse = np.sqrt(mean_squared_error(y[test_idx],prediction))
rSquared = r2_score(y[test_idx],prediction)
print(f"rsme: {rmse} | r^2: {rSquared}")