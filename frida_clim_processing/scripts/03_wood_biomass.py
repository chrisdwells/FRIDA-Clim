import pandas as pd
import numpy as np

start_year = 1850
end_year = 2100

ssps = ['ssp126', 'ssp245', 'ssp370']


df_wood_biomass_hist = pd.read_csv(
    "../data/landuse/hist/hist_LUH_wood_harvest_biomass_for_FRIDA.csv", index_col = 'Unnamed: 0')

df_wood_biomass_hist = df_wood_biomass_hist.loc[df_wood_biomass_hist.index >= start_year]

for scen in ssps:
        
    df_wood_biomass = pd.DataFrame()
    df_wood_biomass['Year'] = np.arange(start_year, end_year+1)
    df_wood_biomass = df_wood_biomass.set_index('Year')

    df_wood_biomass_scen = pd.read_csv(
        f"../data/landuse/{scen}/{scen}_LUH_wood_harvest_biomass_for_FRIDA.csv", index_col = 'Unnamed: 0')
    df_wood_biomass_scen = df_wood_biomass_scen.loc[(df_wood_biomass_scen.index >= 2016)
                                                  & (df_wood_biomass_scen.index <= end_year)]
    df_wood_biomass_full = pd.concat((df_wood_biomass_hist, df_wood_biomass_scen), axis=0)
    
    df_wood_biomass['Forest.cutting'] = df_wood_biomass_full['wood_harvest_biomass']
    df_wood_biomass.to_csv(f'../data/inputs/frida_clim_land_wood_biomass_{scen}.csv')
    