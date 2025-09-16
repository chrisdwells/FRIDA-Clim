import pandas as pd


ssps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

for scen in ssps:
        
    df_emissions = pd.read_csv(f'../data/processed_for_frida/intermediates/emissions_{scen}.csv')
    df_transitions = pd.read_csv(f'../data/processed_for_frida/intermediates/frida_clim_land_transitions_hist_{scen.split("-")[0]}.csv')
    
    df_wood_biomass = pd.read_csv(f'../data/processed_for_frida/intermediates/frida_clim_land_wood_biomass_{scen.split("-")[0]}.csv')
    
    df_crop = pd.read_csv(f'../data/processed_for_frida/intermediates/crop_production_{scen.split("-")[0]}.csv')
    
    df_transitions = df_transitions.drop('Year', axis=1)
    df_wood_biomass = df_wood_biomass.drop('Year', axis=1)
    df_crop = df_crop.drop('Year', axis=1)

    df_full = pd.concat([df_emissions, df_transitions,  df_wood_biomass, df_crop], axis=1)
    
    df_full.to_csv(f'../data/processed_for_frida/ssps/all_inputs_{scen}.csv', index=False)

