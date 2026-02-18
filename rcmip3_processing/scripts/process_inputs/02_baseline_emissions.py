import pandas as pd

indir = '../../../RCMIP3_protocol_bundle/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

baseline_ems_species = {
    'Emissions|CH4': 'Emissions.CH4 Baseline Emissions',
    'Emissions|CO': 'Emissions.Baseline CO Emissions',
    'Emissions|N2O': 'Emissions.N2O Baseline Emissions',
    'Emissions|NOx': 'Emissions.Baseline NOx Emissions',
    'Emissions|Sulfur': 'Emissions.SO2 Baseline Emissions',
    'Emissions|VOC': 'Emissions.Baseline VOC Emissions',
}

data_in = pd.read_csv(f'{indir}/rcmip_phase3_emissions_v1.0.0.csv')

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
