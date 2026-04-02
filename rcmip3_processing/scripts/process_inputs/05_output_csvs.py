import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

conc = pd.read_csv(f"{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv")
emis = pd.read_csv(f"{indir}/rcmip_phase3_emissions_{rcmip_version}.csv")
forc = pd.read_csv(f"{indir}/rcmip_phase3_forcing_{rcmip_version}.csv")

scenarios = pd.concat([
    conc["Scenario"],
    emis["Scenario"],
    forc["Scenario"]
])

expts = sorted(scenarios.unique())
# expts = ['esm-allGHG-hist']

# note this doesnt include the scenariomip ones yet..

#%%
os.makedirs("../../data/frida_clim_output/", exist_ok=True)

for exp in expts:
    csv = f'../../data/frida_clim_output/{exp}.csv'
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

