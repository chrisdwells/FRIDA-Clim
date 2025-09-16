import pandas as pd

start_year = 1750

# initial grassland calculated as residual from total in FRIDA

luh_to_frida = {
    'Land Use.Initial Mature Forest':'forest',
    # 'Land Use.Initial Grassland':'grass',
    'Land Use.Initial Cropland':'cropland',
    'Land Use.Initial Degraded Land':'degraded',
    }

df_hist_stocks = pd.read_csv(
    "../data/external/landuse/hist/hist_LUH_stocks_for_FRIDA.csv", index_col = 'Unnamed: 0')


df_out = pd.DataFrame(columns=luh_to_frida.keys())

row = []
for lu in luh_to_frida.keys():
    
     row.append(df_hist_stocks[luh_to_frida[lu]][start_year])
df_out.loc[0] = row

df_out.to_csv('../data/processed_for_frida/priors_inputs/frida_clim_land_stocks.csv')
