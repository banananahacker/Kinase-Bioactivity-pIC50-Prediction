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
TARGET_MOL = sys.argv[2]


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


mol = Chem.MolFromSmiles(TARGET_MOL)
if mol is None:
    sys.exit(f"Invalid SMILES: {TARGET_MOL}")

fp = fp_gen.GetFingerprintAsNumPy(mol)
descript = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])
features = np.concatenate((fp, descript)).reshape(1, -1) 

print(f"Predicted pIC50 for {TARGET_MOL}: {rf.predict(features)[0]:.2f}")