import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# just makes the ems-driven CO2 emissions files

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_emissions_{rcmip_version}.csv')

scens = [#'esm-pi-cdr-pulse', 'esm-pi-CO2pulse', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC',
         'esm-flat10', 'esm-flat10-zec', 'esm-flat10-cdr', 'esm-flat10-nz', 'esm-flat10-rev', 
         'esm-flat7.5', 'esm-flat7.5-cdr', 'esm-flat7.5-zec', 'esm-flat7.5-nz', 'esm-flat7.5-rev', 
         'esm-flat20', 'esm-flat20-cdr', 'esm-flat20-zec', 'esm-flat20-nz', 'esm-flat20-rev']

#%%

for scen in scens:
        
    in_filtered = data_in[
        (data_in['Scenario'] == scen) &
        (data_in['Region'] == 'World') &
        (data_in['Variable'] == 'Emissions|CO2')
    ]
        
    
    year_cols = [
        c for c in in_filtered.columns
        if c.isdigit() and not pd.isna(in_filtered.iloc[0][c])
    ]
    
    in_long = in_filtered.melt(
        id_vars='Variable',
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    
    in_long['Year'] = in_long['Year'].astype(int)
    
    in_df = (
        in_long
        .pivot(index='Year', columns='Variable', values='Value')
        .sort_index()
    )
        
    
    in_df = in_df.rename(columns={
        'Emissions|CO2': 'Emissions.CO2 Emissions from Fossil use'
    })

    in_df.to_csv(f'{outdir}/{scen}.csv')
    
