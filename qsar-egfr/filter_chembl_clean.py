import sys
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
import math
from concurrent.futures import ProcessPoolExecutor
import os

def _init_worker():
    global _lfc, _uncharger, _te
    RDLogger.DisableLog("rdApp.*")
    _lfc = rdMolStandardize.LargestFragmentChooser()
    _uncharger = rdMolStandardize.Uncharger()
    _te = rdMolStandardize.TautomerEnumerator()

def smi_cleanup_worker(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        clean_mol = rdMolStandardize.Cleanup(mol)
        clean_mol = _lfc.choose(clean_mol)
        clean_mol = _uncharger.uncharge(clean_mol)
        #clean_mol = _te.Canonicalize(clean_mol)
        return Chem.MolToSmiles(clean_mol)
    except Exception:
        return None

def has_carbon(smiles):
    if pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())

def has_ring(smiles):
    if pd.isna(smiles):
        return True
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return True
    return mol.GetRingInfo().NumRings() != 0

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    TARGET = sys.argv[1] if len(sys.argv) > 1 else "CHEMBL203"  
    df = pd.read_csv(f"{TARGET}_raw.csv")
    print(f"Filtering data for {TARGET}")
    print(f"Raw activity records: {len(df)}")

    clean_df = df[~df["assay_description"].str.contains("mutant|L858R|T790M|L861Q|C797S",na=False,case=False)]
    print(f"No mutants in description: {len(clean_df)}")
    clean_df = clean_df[clean_df["assay_type"] == "B"]
    print(f"Assay type B: {len(clean_df)}")
    clean_df = clean_df[clean_df["potential_duplicate"].astype(str).str.contains("0|nan")]
    print(f"No potential duplicates: {len(clean_df)}")
    clean_df = clean_df[clean_df["assay_variant_mutation"].isna()].copy()
    print(f"No assay variant mutations: {len(clean_df)}")

    unique_smiles = clean_df["canonical_smiles"].dropna().unique()
    print(f"Standardizing {len(unique_smiles)} unique SMILES across {os.cpu_count()} cores...")
    with ProcessPoolExecutor(initializer=_init_worker) as executor:
        clean_unique_smiles = list(executor.map(smi_cleanup_worker, unique_smiles, chunksize=100))
    smiles_map = dict(zip(unique_smiles, clean_unique_smiles))
    clean_df["canonical_smiles"] = clean_df["canonical_smiles"].map(smiles_map)

    clean_df = clean_df[clean_df["canonical_smiles"].notna()].copy()
    print(f"Valid standardized SMILES: {len(clean_df)}")

    clean_df = clean_df[clean_df["canonical_smiles"].apply(has_carbon)].copy()
    print(f"No molecules without Carbon atoms: {len(clean_df)}")
    
    repeat_smiles = clean_df.duplicated(subset=["canonical_smiles"]).sum()
    print(f"Extra rows with the same smiles: {repeat_smiles}")
  

    clean_df["standard_value"] = pd.to_numeric(clean_df["standard_value"], errors="coerce")
    clean_df = clean_df[clean_df["standard_value"] > 0].copy()
    clean_df["pIC50"] = 9 - np.log10(clean_df["standard_value"])

    clean_df = clean_df[clean_df["pIC50"] < 12]
    print(f"Rows with pIC50>12 removed: {len(clean_df)}")
    clean_df = clean_df[clean_df["pIC50"] > 3]
    print(f"Rows with pIC50<3 removed: {len(clean_df)}")

    clean_df = clean_df[clean_df["canonical_smiles"].apply(has_ring)].copy()
    print(f"No molecules without rings: {len(clean_df)}")

    clean_df = (
        clean_df.groupby("canonical_smiles")
        .agg(
            mol_chembl_id=("molecule_chembl_id", "first"),
            pIC50=("pIC50", "median"),
            pIC50_std=("pIC50", "std"),
            n_measurements=("pIC50", "size"))
        .reset_index()
        .rename(columns={"canonical_smiles": "smiles"})
    )

    

    out = f"{TARGET}_clean.csv"
    clean_df.to_csv(out, index=False)
    print(f"Unique molecules: {len(clean_df)}")
    print(f"Clean activity records: {len(clean_df)}\n")




