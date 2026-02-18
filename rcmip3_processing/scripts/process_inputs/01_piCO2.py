import pandas as pd

indir = '../../../RCMIP3_protocol_bundle/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_concentrations_v1.0.0.csv')

pi_co2 = data_in[
    (data_in['Model'] == 'unspecified') &
    (data_in['Scenario'] == 'piControl') &
    (data_in['Region'] == 'World') &
    (data_in['Variable'] == 'Atmospheric Concentrations|CO2') &
    (data_in['Unit'] == 'ppm') &
    (data_in['Activity_Id'] == 'input4MIPs') &
    (data_in['Type'] == 'non-idealised') &
    (data_in['Priority'] == '1') &
    (data_in['Mip_Era'] == 'CMIP6') &
    (data_in['Version'] == 'RCMIP Phase 3 v1.0.0')
]['1750'].iloc[0]

df_pi_co2_spinup = pd.DataFrame(
    [pi_co2],  
    columns=["Ocean.Atmospheric CO2 Concentration 1750"]
)

df_pi_co2_spinup.to_csv(f"{outdir}/piControl_CO2_1750.csv", index=False)


df_pi_co2 = pd.DataFrame(
    [pi_co2],  
    columns=["CO2 Forcing.Atmospheric CO2 Concentration 1750"]
)

df_pi_co2.to_csv(f"{outdir}/CO2_1750.csv", index=False)