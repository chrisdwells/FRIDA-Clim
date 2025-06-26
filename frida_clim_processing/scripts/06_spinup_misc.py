import pandas as pd

start_year = 1750

df_crop = pd.read_csv(
    "../data/processed_for_frida/crop_production.csv", index_col = 'Year')
df_cutting = pd.read_csv(
    "../data/processed_for_frida/frida_clim_land_wood_biomass_ssp245.csv", index_col = 'Year')


df_out = pd.DataFrame(columns=['Crop.crop production exogenous', 'Forest.cutting exogenous'])

row = []
    
row.append(df_crop['Crop.crop production exogenous'][start_year])
row.append(df_cutting['Forest.cutting exogenous'][start_year])

df_out.loc[0] = row

df_out.to_csv('../data/processed_for_frida/frida_clim_spinup_misc_constants.csv')
