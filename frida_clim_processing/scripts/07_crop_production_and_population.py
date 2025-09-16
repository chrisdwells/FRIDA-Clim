import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fao_start = 1961
fao_data = pd.read_csv('../data/external/fao_crop_production.csv')

#import glob
# there is no clear trend in the ISIMIP data (local check)

# impact_dir = '../../../isimip-df-agri/data/processed/impacts/corrected/total'

# impact_files = glob.glob(f'{impact_dir}/*historical*.csv')

# fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# for f in impact_files:
    
#     data_in = pd.read_csv(f)
    
#     ax[0].plot(data_in['YEARS'], data_in['WORLD']/data_in.loc[data_in['YEARS'] == 1861]['WORLD'].values[0], color='grey')
#     ax[1].plot(data_in['YEARS'], data_in['WORLD']/data_in.loc[data_in['YEARS'] == fao_start]['WORLD'].values[0], color='grey')

    
# ax[0].set_title('Total Production cf 1861')
# ax[1].set_title('Total Production cf 1961')

# ax[1].plot(fao_data['Year'], fao_data['Crop production [GtC]']/fao_data.loc[fao_data['Year'] == fao_start]['Crop production [GtC]'].values[0], color='black', label='FAO')
# ax[1].legend()


#%%

# so use FAO data from 1961, and scale by population before this

hist_pop_csv = pd.read_csv('../data/external/population.csv')  # https://ourworldindata.org/grapher/population
hist_pop = hist_pop_csv.loc[hist_pop_csv['Entity'] == 'World']
hist_pop = hist_pop.drop(columns = ['Entity', 'Code'])
hist_pop = hist_pop.set_index('Year')

pop_crop_scale = fao_data.loc[fao_data['Year'] == fao_start
                  ]['Crop production [GtC]'].values[0] / hist_pop.loc[hist_pop.index == fao_start].values[0][0]

pop_scaled = hist_pop*pop_crop_scale

pop_scaled_crop = pop_scaled.loc[pop_scaled.index >= 1750]

plt.plot(fao_data['Year'], fao_data['Crop production [GtC]'], label='FAO')

plt.plot(pop_scaled_crop.index, pop_scaled_crop, label='Population scaled')
plt.ylabel('GtC')


combined_data = np.concatenate((pop_scaled_crop.loc[pop_scaled_crop.index < fao_start].values[:,0],
                                fao_data['Crop production [GtC]'].values)).astype('float64')


combined_idx = np.concatenate((pop_scaled_crop.loc[pop_scaled_crop.index < fao_start].index,
                                fao_data['Year'].values))

time = np.arange(1750, 2021, 1)

combined_data_interp = np.interp(time, combined_idx, combined_data)

d = {
    'Crop.crop production exogenous': pd.Series(combined_data_interp, index = time)
    }

df = pd.DataFrame(d)
df['Year'] = time
df = df.set_index('Year')

plt.plot(df.index, df['Crop.crop production exogenous'].values, label='Combined', linestyle='--')

plt.legend()

#%%

df.to_csv('../data/processed_for_frida/priors_inputs/crop_production.csv')

#%%

pop_interp = np.interp(time, hist_pop.index.values, hist_pop['Population (historical)'].values*1E-6) # p to Mp

d = {
    'Sea Level.Global population exogenous': pd.Series(pop_interp, index = time)
    }

df = pd.DataFrame(d)
df['Year'] = time
df = df.set_index('Year')

plt.plot(df.index, df['Sea Level.Global population exogenous'].values, label='Population (Mp)')

plt.legend()

df.to_csv('../data/processed_for_frida/priors_inputs/population.csv')

