import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_forcing_{rcmip_version}.csv')

forc_species_from_rcmip = {
    'Effective Radiative Forcing|Natural|Solar': 'Natural Forcing.Baseline Effective Radiative Forcing from Solar Output Variations',
    'Effective Radiative Forcing|Natural|Volcanic': 'Natural Forcing.Baseline Effective Radiative Forcing from Volcanoes',
}


forc_filtered = data_in[
    (data_in['Scenario'] == 'historical') &
    (data_in['Region'] == 'World') &
    (data_in['Variable'].isin(forc_species_from_rcmip.keys()))
]

year_cols = [c for c in forc_filtered.columns if c.isdigit()]

forc_long = forc_filtered.melt(
    id_vars='Variable',
    value_vars=year_cols,
    var_name='Year',
    value_name='Value'
)

forc_long['Year'] = forc_long['Year'].astype(int)
forc_long
forc_df = (
    forc_long
    .pivot(index='Year', columns='Variable', values='Value')
    .sort_index()
)

forc_df = forc_df.loc[1750:2500]

forc_df = forc_df.rename(columns=forc_species_from_rcmip)


forc_df.to_csv(f'{outdir}/natural_forcings.csv')

