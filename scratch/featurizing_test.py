from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import numpy as np

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
mw = Descriptors.MolWt(mol)
print(mol)
print(mw)

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

mol_fped = fp_gen.GetFingerprintAsNumPy(mol)
print(mol_fped)
print(len(mol_fped))
print(np.sum(mol_fped))

mol_descript = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])

mol_fped = np.concatenate((mol_fped, mol_descript))
print(len(mol_fped))
print(mol_fped[-3:])