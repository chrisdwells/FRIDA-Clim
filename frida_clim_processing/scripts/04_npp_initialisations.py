import numpy as np
import pandas as pd

# calculate initial NPP values in start year; take decadal averages

# use 1850-60 due to climate data limits; should be ok but could always estimate further

# just use central value from FRIDA parameters..

# IGCC climate data 
# https://github.com/ClimateIndicator/forcing-timeseries/blob/main/output/ghg_concentrations_1750-2023.csv
# https://github.com/ClimateIndicator/data/blob/main/data/global_mean_temperatures/annual_averages.csv

base_year = 1980
target_year = 1855


df_gmst = pd.read_csv('../data/external/IGCC/annual_averages.csv', index_col=0)

delT = np.mean(df_gmst.loc[(base_year - 5 < df_gmst['timebound_lower']) & 
                   (df_gmst['timebound_lower'] < base_year + 5)]['gmst'].values
       ) - np.mean(df_gmst.loc[(target_year - 5 < df_gmst['timebound_lower']) & 
                   (df_gmst['timebound_lower'] < target_year + 5)]['gmst'].values)
                               

df_conc = pd.read_csv('../data/external/IGCC/ghg_concentrations_1750-2023.csv', index_col=0)

delC = np.mean(df_conc.loc[(base_year - 5 < df_conc.index) & 
                   (df_conc.index < base_year + 5)]['CO2'].values
       ) - np.mean(df_conc.loc[(target_year - 5 < df_conc.index) & 
                   (df_conc.index < target_year + 5)]['CO2'].values)
                                       

# forest
fr_npp_base = 0.008320717	# 0.00828951 to 0.008351924

a = 0.125348729	# 0.120697458	 to 0.13
b = -0.023015026	# -0.03465362	 to -0.011376432
c = 0.001	# 0.001 to 0.001

factor = 1 + a*delT + b*delT**2 + c*delC

fr_npp_target = fr_npp_base/factor


# grass
g_npp_base = 0.005	# 0.005 to 0.005

a = 0.126281692	# 0.122563384	 to 0.13
b = -0.034654395	# -0.05 to -0.01930879
c = 0.000963654	# 0.000927308	 to 0.001

factor = 1 + a*delT + b*delT**2 + c*delC

g_npp_target = g_npp_base/factor

#%%

frida_vars = {
    "Forest.tree net primary production in 1750":fr_npp_target,
    "Grass.grass net primary production in 1750":g_npp_target,
    }

df_out = pd.DataFrame(columns=frida_vars.keys())

row = []
for lu in frida_vars.keys():
    
     row.append(frida_vars[lu])
df_out.loc[0] = row

df_out.to_csv('../data/processed_for_frida/frida_clim_npp_initial_values.csv')
