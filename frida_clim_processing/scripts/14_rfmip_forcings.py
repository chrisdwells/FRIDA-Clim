import pandas as pd

# just processes the RFMIP SSP forcing files to drive exog model

scens = {
    "ssp119":'a',
    "ssp245":'c',
    "ssp534-over":'x',
    "ssp585":'e',
    }

for scen in scens.keys():
    
    df = pd.read_csv(
        f'../../calibration/data/external/forcing/ssp_forcings/table_A3.4{scens[scen]}_{scen}_ERF_1750-2500_best_estimate.csv')

    df_out = df['total']
    df_out = df[['year', 'total']].rename(columns = {
        'year':'Year',
        'total':'Forcing.Exogenous Effective Radiative Forcing'}
        )
    
    df_out.to_csv(f'../data/processed_for_frida/ssps/rfmip_{scen}.csv', index=False)
    
