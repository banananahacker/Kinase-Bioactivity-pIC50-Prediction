import sys
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

TARGET = sys.argv[1] if len(sys.argv) > 1 else "CHEMBL203"  # CHEMBL203 = EGFR


records = new_client.activity.filter(
    target_chembl_id=TARGET,
    standard_type="IC50",
    standard_relation="=",   # drop ">10000"-style censored values
    standard_units="nM",
    #assay_type="B"
)#.only(["molecule_chembl_id", "canonical_smiles", "standard_value", "assay_chembl_id"])

df = pd.DataFrame.from_records(records)
print(f"Raw activity records for {TARGET}: {len(df)}")

out = f"{TARGET}_raw.csv"
df.to_csv(out, index=False)
print(f"Exported {len(df)} raw molecules for {TARGET}")

