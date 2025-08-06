import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# taken from calibrate-FRIDA-climate

# Adapted from FaIR calibrate

# Exclude priors which don't closely match GMST and air-sea CO2 flux

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))

weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

npp_2000_obs = 59.22

#%%

df_temp = pd.read_csv("../data/priors_output/priors_temperature.csv")

temp_hist = df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=2022)].drop(columns='Year').values
temp_hist_offset = temp_hist - np.average(temp_hist[:52, :], weights=weights, axis=0)


df_temp_obs = pd.read_csv("../data/external/forcing/annual_averages.csv")
gmst = df_temp_obs["gmst"].loc[(df_temp_obs['time'] > 1850) 
                               & (df_temp_obs['time'] < 2023)].values

time = df_temp_obs["time"].loc[(df_temp_obs['time'] > 1850) 
                               & (df_temp_obs['time'] < 2023)].values


df_flux = pd.read_csv("../data/priors_output/priors_ocean_CO2_flux.csv")

if '="Calibration Data: Ocean.Air sea co2 flux[1]"' in df_flux.keys(): # this occured once - not sure why
    df_flux = df_flux.drop(['="Calibration Data: Ocean.Air sea co2 flux[1]"'], axis=1)

flux_hist = df_flux.loc[(df_flux['Year']>=1781) & (df_flux['Year']<=2022)].drop(columns='Year').values
flux_hist_for_rmse = df_flux.loc[(df_flux['Year']>=1960) & (df_flux['Year']<=2022)].drop(columns='Year').values


df_ocean = pd.read_csv("../data/external/GCB_historical_budget.csv")
df_ocean_hist = df_ocean.loc[(df_ocean['Year']>=1781) & (df_ocean['Year']<=2022)]

flux = df_ocean_hist["ocean sink"].values

df_ocean_hist_crop = df_ocean.loc[(df_ocean['Year']>=1960) & (df_ocean['Year']<=2022)]
flux_for_rmse = df_ocean_hist_crop["ocean sink"].values


#%%

rmse_temp = np.zeros((samples))

for i in range(samples):
    rmse_temp[i] = rmse(
        gmst,
        temp_hist_offset[:, i],
    )
    
accept_temp = rmse_temp < 0.16

n_pass_temp = np.sum(accept_temp)

print("Passing Temperature constraint:", n_pass_temp)
valid_temp = np.arange(samples, dtype=int)[accept_temp]

flux_constraint = 0.2*np.mean(flux_for_rmse)

rmse_flux = np.zeros((samples))

for i in range(samples):
    rmse_flux[i] = rmse(
        flux_for_rmse[:170],
        flux_hist_for_rmse[:170, i],
    )
    

accept_flux = rmse_flux < flux_constraint

n_pass_flux = np.sum(accept_flux)

print("Passing Flux constraint:",n_pass_flux)
valid_flux = np.arange(samples, dtype=int)[accept_flux]

valid_both = np.intersect1d(valid_temp,valid_flux)

n_pass_both = valid_both.shape[0]

print("Passing both constraints:",n_pass_both)

accept_both = np.logical_and(accept_temp, accept_flux)

#%%
priors_temp_2005_14 = np.mean(df_temp.loc[(df_temp['Year']>=2005) & (
                df_temp['Year']<=2014)].drop(columns='Year').values - np.average(temp_hist[:52, :], 
                     weights=weights, axis=0), axis=0)

priors_flux_2005_14 = np.mean(df_flux.loc[(df_flux['Year']>=2005) & (
                df_flux['Year']<=2014)].drop(columns='Year').values, axis=0)

priors_flux_2005_14_obs = np.mean(df_ocean["ocean sink"].loc[(df_ocean['Year']>=2005) & (
                df_ocean['Year']<=2014)].drop(columns='Year').values, axis=0)


priors_temp_2005_14_obs = np.mean(df_temp_obs["gmst"].values[-16:-6])


plt.scatter(priors_temp_2005_14, priors_flux_2005_14, color='grey')

plt.axhline(y = priors_flux_2005_14_obs)
plt.axhline(y = priors_flux_2005_14_obs+flux_constraint)
plt.axhline(y = priors_flux_2005_14_obs-flux_constraint)

plt.axvline(x = priors_temp_2005_14_obs)
plt.axvline(x = priors_temp_2005_14_obs+0.16)
plt.axvline(x = priors_temp_2005_14_obs-0.16)


plt.xlabel('Temp 2005-14')
plt.ylabel('Flux 2005-14')


#%%

df_npp = pd.read_csv("../data/priors_output/priors_NPP.csv")


if '="Calibration Data: Terrestrial Carbon Balance.Terrestrial net primary production[1]"' in df_npp.keys(): # this occured once - not sure why
    df_npp = df_npp.drop(['="Calibration Data: Terrestrial Carbon Balance.Terrestrial net primary production[1]"'], axis=1)



npp_2000 = np.full(samples, np.nan)
for i in np.arange(samples):
    npp_2000[i] = df_npp[f'="Run {i+1}: Terrestrial Carbon Balance.Terrestrial net primary production[1]"']


npp_grass_2000 = np.full(samples, np.nan)
for i in np.arange(samples):
    npp_grass_2000[i] = df_npp[f'="Run {i+1}: Grass.grassland net primary production[1]"']

npp_constraint = npp_2000_obs    

accept_npp = np.abs(npp_2000[:] -  npp_constraint) < 10

n_pass_npp = np.sum(accept_npp)

print("Passing NPP constraint:",n_pass_npp)
valid_npp = np.arange(samples, dtype=int)[accept_npp]

valid_inc_npp = np.intersect1d(valid_both,valid_npp)

n_pass_inc_npp = valid_inc_npp.shape[0]

print("Passing inc NPP:",n_pass_inc_npp)

accept_inc_npp = np.logical_and(accept_both, accept_npp)

#%%

fig, axs = plt.subplots(5, 2, figsize=(12, 15))

axs[0,0].fill_between(time, np.percentile(temp_hist_offset, 84, axis=1), 
              np.percentile(temp_hist_offset, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,0].plot(time, np.median(temp_hist_offset, axis=1), 
              color="#000000", label='Median')

axs[0,0].plot(time, gmst, label='AR6 obs')

axs[0,0].legend()
axs[0,0].set_ylabel('deg C')
axs[0,0].set_title(f'All priors: {samples}')



axs[0,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist, 84, axis=1), 
              np.percentile(flux_hist, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,1].plot(df_ocean_hist["Year"], np.median(flux_hist, axis=1), 
              color="#000000", label='Median')

axs[0,1].plot(df_ocean_hist["Year"], flux)
axs[0,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[0,1].legend()
axs[0,1].set_ylabel('GtC/yr')
axs[0,1].set_title(f'All priors: {samples}')





axs[1,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_temp], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,0].plot(time, np.median(temp_hist_offset[:, accept_temp], axis=1), 
              color="#000000", label='Median')

axs[1,0].plot(time, gmst, label='AR6 obs')

axs[1,0].legend()
axs[1,0].set_ylabel('deg C')
axs[1,0].set_title(f'Passing temp: {n_pass_temp}')



axs[1,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_temp], 84, axis=1), 
              np.percentile(flux_hist[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_temp], axis=1), 
              color="#000000", label='Median')

axs[1,1].plot(df_ocean_hist["Year"], flux)
axs[1,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[1,1].legend()
axs[1,1].set_ylabel('GtC/yr')
axs[1,1].set_title(f'Passing temp: {n_pass_temp}')




axs[2,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_flux], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,0].plot(time, np.median(temp_hist_offset[:, accept_flux], axis=1), 
              color="#000000", label='Median')

axs[2,0].plot(time, gmst, label='AR6 obs')

axs[2,0].legend()
axs[2,0].set_ylabel('deg C')
axs[2,0].set_title(f'Passing flux: {n_pass_flux}')



axs[2,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_flux], 84, axis=1), 
              np.percentile(flux_hist[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_flux], axis=1), 
              color="#000000", label='Median')

axs[2,1].plot(df_ocean_hist["Year"], flux)
axs[2,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[2,1].legend()
axs[2,1].set_ylabel('GtC/yr')
axs[2,1].set_title(f'Passing flux: {n_pass_flux}')





axs[3,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_both], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,0].plot(time, np.median(temp_hist_offset[:, accept_both], axis=1), 
              color="#000000", label='Median')

axs[3,0].plot(time, gmst, label='AR6 obs')

axs[3,0].legend()
axs[3,0].set_ylabel('deg C')
axs[3,0].set_title(f'Passing both: {n_pass_both}')



axs[3,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_both], 84, axis=1), 
              np.percentile(flux_hist[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_both], axis=1), 
              color="#000000", label='Median')

axs[3,1].plot(df_ocean_hist["Year"], flux)
axs[3,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[3,1].legend()
axs[3,1].set_ylabel('GtC/yr')
axs[3,1].set_title(f'Passing both: {n_pass_both}')



axs[4,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_inc_npp], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_inc_npp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[4,0].plot(time, np.median(temp_hist_offset[:, accept_inc_npp], axis=1), 
             color="#000000", label='Median')

axs[4,0].plot(time, gmst, label='AR6 obs')

axs[4,0].legend()
axs[4,0].set_ylabel('deg C')
axs[4,0].set_title(f'Passing inc NPP 2000: {n_pass_inc_npp}')



axs[4,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_inc_npp], 84, axis=1), 
              np.percentile(flux_hist[:, accept_inc_npp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[4,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_inc_npp], axis=1), 
              color="#000000", label='Median')

axs[4,1].plot(df_ocean_hist["Year"], flux)
axs[4,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[4,1].legend()
axs[4,1].set_ylabel('GtC/yr')
axs[4,1].set_title(f'Passing inc NPP 2000: {n_pass_inc_npp}')

plt.tight_layout()

os.makedirs("../plots", exist_ok=True)

plt.savefig(
    "../plots/rmse_constrained.png"
)

#%%
import scipy.stats

npp_priors = scipy.stats.gaussian_kde(npp_2000)


plt.plot(np.linspace(40, 75, 1000), npp_priors(np.linspace(40, 75, 1000)))
plt.axvline(x=npp_2000_obs)
plt.axvline(x=npp_2000_obs-10, linestyle='--')
plt.axvline(x=npp_2000_obs+10, linestyle='--')
plt.title('NPP')


#%%
np.savetxt(
    "../data/constraining/runids_rmse_pass.csv",
    valid_both.astype(int),
    fmt="%d",
)

#%%


# TO DO 
df_land = pd.read_csv("../data/priors_output/priors_land.csv")


#%%

gcb = pd.read_csv("../../../calibrate-FRIDA-climate/data/external/gcp_v2023_co2_1750-2022.csv")

land_sink_obs = gcb['land sink'].values[:273]

tcb_obs = gcb['terrestrial carbon balance'].values[:273]

gcb_time = 1750 + np.arange(273)

tcb_data = np.full((273, samples), np.nan)
for i in np.arange(samples):
    tcb_data[:,i] = df_land[f'="Run {i+1}: Terrestrial Carbon Balance.Terrestrial carbon balance[1]"']

landsink_data = np.full((273, samples), np.nan)
for i in np.arange(samples):
    landsink_data[:,i] = df_land[f'="Run {i+1}: Terrestrial Carbon Balance.Terrestrial carbon balance[1]"'] + df_land[
                                f'="Run {i+1}: Emissions.CO2 Emissions from Food and Land Use[1]"']/3670 # MtCO2 to GtC
        #%%

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    

axs[0].fill_between(gcb_time, np.percentile(tcb_data[:, accept_inc_npp], 84, axis=1), 
              np.percentile(tcb_data[:, accept_inc_npp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0].plot(gcb_time, np.median(tcb_data[:, accept_inc_npp], axis=1), 
             color="#000000", label='Median')

axs[0].plot(gcb_time, tcb_obs, label='GCB')

axs[0].legend()
axs[0].set_ylabel('GtC')
axs[0].set_title('TCB')


axs[1].fill_between(gcb_time, np.percentile(landsink_data[:, accept_inc_npp], 84, axis=1), 
              np.percentile(landsink_data[:, accept_inc_npp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1].plot(gcb_time, np.median(landsink_data[:, accept_inc_npp], axis=1), 
             color="#000000", label='Median')

axs[1].plot(gcb_time, land_sink_obs, label='GCB')

axs[1].legend()
axs[1].set_ylabel('GtC')
axs[1].set_title('Land sink')

plt.tight_layout()
