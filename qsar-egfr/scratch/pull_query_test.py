import sys
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*") 

TARGET = sys.argv[1] if len(sys.argv) > 1 else "CHEMBL203"  
'''
print(f"Querying ChEMBL for {TARGET} (can take a few minutes)...")
records = new_client.activity.filter(
    target_chembl_id=TARGET,
    standard_type="IC50",
    standard_relation="=",   
    standard_units="nM",
    assay_type="B",
    assay_variant_mutation__isnull = True
).only(["molecule_chembl_id", "canonical_smiles", "standard_value", "assay_chembl_id", "assay_type", "data_validity_comment","potential_duplicate", "assay_variant_mutation", "target_organism", "bao_format","assay_description"])

df = pd.DataFrame.from_records(records)
'''

df = pd.read_csv("CHEMBL203_raw.csv")
print(f"Raw activity records: {len(df)}")

clean_df = df[~df["assay_description"].str.contains("mutant|L858R|T790M|L861Q|C797S",na=False,case=False)]
print(f"No mutants in description: {len(clean_df)}")
clean_df = clean_df[clean_df["assay_type"] == "B"]
print(f"Assay type B: {len(clean_df)}")
clean_df = clean_df[clean_df["potential_duplicate"].astype(str).str.contains("0|nan")]
print(f"No potential duplicates: {len(clean_df)}")
clean_df = clean_df[clean_df["assay_variant_mutation"].isna()]
print(f"No assay variant mutations: {len(clean_df)}")

repeat_ids = clean_df.duplicated(subset=["molecule_chembl_id"]).sum()
print(f"Extra rows with the same molecule_chembl_id: {repeat_ids}")

print(f"Clean activity records: {len(clean_df)}")
out = f"{TARGET}_clean.csv"
clean_df.to_csv(out, index=False)
