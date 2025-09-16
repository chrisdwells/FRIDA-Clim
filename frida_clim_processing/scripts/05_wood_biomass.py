import pandas as pd
import numpy as np

# just make to 2100; then Stella will extend at this value

start_year = 1750
end_year_dynamics = 2099
# end_year = 2500
hist_end_year = 2022

ssps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534', 'ssp585']


df_wood_biomass_hist = pd.read_csv(
    "../data/external/landuse/hist/hist_LUH_wood_harvest_biomass_for_FRIDA.csv", index_col = 'Unnamed: 0')

df_wood_biomass_hist = df_wood_biomass_hist.loc[df_wood_biomass_hist.index >= start_year]

hist_end = df_wood_biomass_hist.index[-1]

for scen in ssps:
        
    df_wood_biomass = pd.DataFrame()
    df_wood_biomass['Year'] = np.arange(start_year, end_year_dynamics+1)
    df_wood_biomass = df_wood_biomass.set_index('Year')

    df_wood_biomass_scen = pd.read_csv(
        f"../data/external/landuse/{scen}/{scen}_LUH_wood_harvest_biomass_for_FRIDA.csv", index_col = 'Unnamed: 0')
    
    if hist_end+1 not in df_wood_biomass_scen.index:
        raise Exception('Historical and scenario do not line up!')
    
    df_wood_biomass_ssp = df_wood_biomass_scen.loc[(df_wood_biomass_scen.index > hist_end)
                                                  & (df_wood_biomass_scen.index <= end_year_dynamics)]
    
    
    df_wood_biomass_full = pd.concat((df_wood_biomass_hist, df_wood_biomass_ssp), axis=0)
    
    df_wood_biomass['Forest.cutting exogenous'] = df_wood_biomass_full['wood_harvest_biomass']
    df_wood_biomass.to_csv(f'../data/processed_for_frida/intermediates/frida_clim_land_wood_biomass_{scen}.csv')
    
    
    if scen == 'ssp245':
        df_wood_biomass_only = df_wood_biomass.loc[start_year:hist_end_year]
        df_wood_biomass_only.to_csv('../data/processed_for_frida/priors_inputs/frida_clim_land_wood_biomass_hist.csv')

