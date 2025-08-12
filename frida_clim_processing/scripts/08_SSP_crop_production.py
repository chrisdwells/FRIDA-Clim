import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

common_period = [2010,2020]

df_hist = pd.read_csv('../data/processed_for_frida/crop_production.csv')

hist_end_year = df_hist['Year'].values[-1]

crop_common_period = np.mean(df_hist.loc[(common_period[0] <= df_hist['Year'
              ]) & (df_hist['Year'] <= common_period[1])]['Crop.crop production exogenous'])

ssps = {
        'ssp119':'SSP1-19', 
        'ssp126':'SSP1-26', 
        'ssp245':'SSP2-45', 
        'ssp370':'SSP3-Baseline', 
        'ssp434':'SSP4-34', 
        'ssp460':'SSP4-60', 
        'ssp534':'SSP5-34', 
        'ssp585':'SSP5-Baseline',
        }

df_energy = pd.read_csv('../data/external/ssps_crop_production/iamc_db_energy.csv')
df_nonenergy = pd.read_csv('../data/external/ssps_crop_production/iamc_db_nonenergy.csv')

#%%
iam_years = [2005, 2010, 2020, 2030, 2050, 2060, 2070, 2080, 2090, 2100]
interp_years = np.arange(2005, 2100+1, 1)

idx_first_future_year = int(np.where(interp_years == hist_end_year)[0]) + 1

common_period_idxs = [int(np.where(interp_years == yr)[0]) for yr in common_period]

combined_time = np.concat((df_hist['Year'].values, interp_years[idx_first_future_year:]))

plt.plot(df_hist['Year'], df_hist['Crop.crop production exogenous'])  


for scen in ssps:

    df_out = pd.DataFrame(columns = ['Year', 'Crop.crop production exogenous'])
    
    df_out['Year'] = combined_time

    scen_data = []
    for y_i, year in enumerate(iam_years):
        
        data_energy = df_energy.loc[df_energy['Scenario'] == ssps[scen]][f'{year}'].values[0]
        data_nonenergy = df_nonenergy.loc[df_nonenergy['Scenario'] == ssps[scen]][f'{year}'].values[0]

        scen_data.append(data_energy+data_nonenergy)
        
    scen_data_interp = np.interp(interp_years, np.asarray(iam_years), scen_data)
    
    scen_data_interp_common = np.mean(scen_data_interp[common_period_idxs[0]:common_period_idxs[1]+1])

    scen_data_interp_scaled = scen_data_interp*(crop_common_period/scen_data_interp_common)
    
    hist_scen_combined = np.concat((df_hist['Crop.crop production exogenous'], scen_data_interp_scaled[idx_first_future_year:]))
    
    plt.plot(interp_years, scen_data_interp_scaled, label = f'{scen}')
    
    plt.plot(combined_time, hist_scen_combined, label = f'Hist+{scen}')
    
    df_out['Crop.crop production exogenous'] = hist_scen_combined

    df_out.to_csv(f'../data/processed_for_frida/crop_production_{scen}.csv', index=False)

plt.legend(ncol=2)
    