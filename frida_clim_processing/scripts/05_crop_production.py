import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

impact_dir = '../../../isimip-df-agri/data/processed/impacts/corrected/total'

impact_files = glob.glob(f'{impact_dir}/*historical*.csv')

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

for f in impact_files:
    
    data_in = pd.read_csv(f)
    
    ax[0].plot(data_in['YEARS'], data_in['WORLD']/data_in.loc[data_in['YEARS'] == 1861]['WORLD'].values[0], color='grey')
    ax[1].plot(data_in['YEARS'], data_in['WORLD']/data_in.loc[data_in['YEARS'] == 1961]['WORLD'].values[0], color='grey')

    
ax[0].set_title('Total Production cf 1861')
ax[1].set_title('Total Production cf 1961')

fao_data = pd.read_csv('../data/inputs/fao_crop_production.csv')
ax[1].plot(fao_data['Year'], fao_data['Crop production [GtC]']/fao_data.loc[fao_data['Year'] == 1961]['Crop production [GtC]'].values[0], color='black', label='FAO')
ax[1].legend()


#%%

gtc_per_pcal = 0.000220559

frida_calib = pd.read_csv('../../../WorldTransFRIDA/Data/Calibration Data.csv', index_col=0)

frida_calib = frida_calib.T

frida_calib_hist = frida_calib['1980':'2020']

frida_calib_crop = frida_calib_hist['Crop.Crop Production[1]']
frida_calib_pop = frida_calib_hist['Demographics.Population[1]']

idx_num = [int(idx) for idx in frida_calib_hist.index]

plt.plot(idx_num, frida_calib_crop*gtc_per_pcal, label='FRIDA calib (FAO)')

hist_pop_csv = pd.read_csv('../data/inputs/population.csv')  # https://ourworldindata.org/grapher/population
hist_pop = hist_pop_csv.loc[hist_pop_csv['Entity'] == 'World']
hist_pop = hist_pop.drop(columns = ['Entity', 'Code'])
hist_pop = hist_pop.set_index('Year')

pop_crop_scale = frida_calib_crop.loc[frida_calib_crop.index == '1980'
                      ].values[0] / hist_pop.loc[hist_pop.index == 1980].values[0][0]

pop_scaled = hist_pop*pop_crop_scale

pop_scaled_crop = pop_scaled.loc[pop_scaled.index >= 1750]

plt.plot(pop_scaled_crop.index, pop_scaled_crop*gtc_per_pcal, label='Population scaled')
plt.ylabel('GtC')




combined_data = gtc_per_pcal*np.concatenate((pop_scaled_crop.loc[pop_scaled_crop.index < 1980].values[:,0],
                                frida_calib_crop.values)).astype('float64')


combined_idx = np.concatenate((pop_scaled_crop.loc[pop_scaled_crop.index < 1980].index,
                                idx_num))

time = np.arange(1750, 2021, 1)

combined_data_interp = np.interp(time, combined_idx, combined_data)

d = {
    'Crop Production': pd.Series(combined_data_interp, index = time)
    }

df = pd.DataFrame(d)

plt.plot(df.index, df['Crop Production'].values, label='Combined', linestyle='--')

plt.legend()

df.to_csv('../data/inputs/crop_production.csv')

