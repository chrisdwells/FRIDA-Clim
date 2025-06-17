import pandas as pd
import numpy as np

start_year = 1850
end_year = 2100

ssps = ['ssp126', 'ssp245', 'ssp370']


# in v2.1, we don't simulate some transitions - leave out for now
luh_to_frida = {
    'forest_to_grass':'Land Use.clearing forest for grassland exogenous',	
    'forest_to_cropland':'Land Use.clearing forests for crops exogenous',
    # 'forest_to_degraded':'Land Use.',
    'grass_to_forest':'Land Use.foresting grassland exogenous',
    'grass_to_cropland':'Land Use.clearing grassland exogenous',
    'grass_to_degraded':'Land Use.degraded land from grassland exogenous',
    'cropland_to_forest':'Land Use.forresting cropland exogenous',
    'cropland_to_grass':'Land Use.fallowing cropland exogenous',
    'cropland_to_degraded':'Land Use.degraded land from crops exogenous',
    'degraded_to_forest':'Land Use.foresting degraded land exogenous',
    # 'degraded_to_grass':'Land Use.',
    # 'degraded_to_cropland':'Land Use.',
    }


df_transitions_hist = pd.read_csv(
    "../data/landuse/hist/hist_LUH_transitions_for_FRIDA.csv", index_col = 'Unnamed: 0')

df_transitions_hist = df_transitions_hist.loc[df_transitions_hist.index >= start_year]

for scen in ssps:
        
    df_transitions = pd.DataFrame()
    df_transitions['Year'] = np.arange(start_year, end_year+1)
    df_transitions = df_transitions.set_index('Year')

    df_transitions_scen = pd.read_csv(
        f"../data/landuse/{scen}/{scen}_LUH_transitions_for_FRIDA.csv", index_col = 'Unnamed: 0')
    df_transitions_scen = df_transitions_scen.loc[df_transitions_scen.index <= end_year]
    
    df_transitions_full = pd.concat((df_transitions_hist, df_transitions_scen), axis=0)
    
    for transition in luh_to_frida.keys():
        df_transitions[luh_to_frida[transition]] = df_transitions_full[transition]
    
    df_transitions.to_csv(f'../data/inputs/frida_clim_land_transitions_hist_{scen}.csv')
    
