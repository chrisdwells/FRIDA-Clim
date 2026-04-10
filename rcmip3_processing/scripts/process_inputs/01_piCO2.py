import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv')

baseline_species = {
    'CO2':'ppm', 
    'N2O':'ppb', 
    'CH4':'ppb',
    }

spec_names = []
data_out = []
for spec in baseline_species.keys():
    pi = data_in[
        (data_in['Model'] == 'emissions_harmonisation_pipeline') &
        (data_in['Scenario'] == 'piControl') &
        (data_in['Region'] == 'World') &
        (data_in['Variable'] == f'Atmospheric Concentrations|{spec}') &
        (data_in['Unit'] == baseline_species[spec]) &
        (data_in['Activity_Id'] == 'input4MIPs') &
        (data_in['Type'] == 'non-idealised') &
        (data_in['Priority'] == '1') &
        (data_in['Mip_Era'] == 'CMIP7') &
        (data_in['Version'] == f'RCMIP Phase 3 {rcmip_version}')
    ]['1750'].iloc[0]
    
    if spec == 'CO2':
        df_pi_co2_spinup = pd.DataFrame(
            [pi],  
            columns=["Ocean.Atmospheric CO2 Concentration 1750"]
        )
        df_pi_co2_spinup.to_csv(f"{outdir}/piControl_CO2_1750.csv", index=False)
    
    spec_names.append(f'{spec} Forcing.Atmospheric {spec} Concentration 1750')
    data_out.append(f'{pi}')

df_pi = pd.DataFrame(
    [data_out],  
    columns=spec_names
)

df_pi.to_csv(f"{outdir}/concs_1750.csv", index=False)