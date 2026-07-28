import pandas as pd

outdir = '../../data/processed_for_frida'

# copied over from prior FRIDA version - as we want to update these columns
old = pd.read_csv("../../data/external/frida_input_data_prior.csv")
new = pd.read_csv(f"{outdir}/natural_forcings.csv")

cols = [
    "Natural Forcing.Baseline Effective Radiative Forcing from Solar Output Variations",
    "Natural Forcing.Baseline Effective Radiative Forcing from Volcanoes"
]

new_sub = new[new["Year"].between(1980, 2150)]

for col in cols:
    old.loc[old["Year"].between(1980, 2150), col] = new_sub[col].values

old.to_csv("../../data/external/frida_input_data.csv", index=False)