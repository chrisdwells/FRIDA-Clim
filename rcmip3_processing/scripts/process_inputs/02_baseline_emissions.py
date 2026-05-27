import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

baseline_ems_species = {
    'Emissions|CH4': 'Emissions.CH4 Baseline Emissions',
    'Emissions|CO': 'Emissions.Baseline CO Emissions',
    'Emissions|N2O': 'Emissions.N2O Baseline Emissions',
    'Emissions|NOx': 'Emissions.Baseline NOx Emissions',
    'Emissions|Sulfur': 'Emissions.SO2 Baseline Emissions',
    'Emissions|VOC': 'Emissions.Baseline VOC Emissions',
}

data_in = pd.read_csv(f'{indir}/rcmip_phase3_emissions_{rcmip_version}.csv')

ems_filtered = data_in[
    (data_in['Scenario'] == 'historical') &
    (data_in['Region'] == 'World')
]

values_1750 = [
    ems_filtered.loc[
        ems_filtered['Variable'] == var, '1750'
    ].iloc[0]
    for var in baseline_ems_species.keys()
]

ems_df = pd.DataFrame(
    [values_1750],
    columns=list(baseline_ems_species.values())
)

ems_df = ems_df.astype(float)

ems_df.to_csv(f'{outdir}/baseline_emissions.csv', index=False)


ems_filtered = data_in[
    (data_in['Scenario'] == 'historical-cmip6') &
    (data_in['Region'] == 'World')
]

values_1750 = [
    ems_filtered.loc[
        ems_filtered['Variable'] == var, '1750'
    ].iloc[0]
    for var in baseline_ems_species.keys()
]

ems_df = pd.DataFrame(
    [values_1750],
    columns=list(baseline_ems_species.values())
)

ems_df = ems_df.astype(float)

ems_df.to_csv(f'{outdir}/baseline_emissions_cmip6.csv', index=False)
