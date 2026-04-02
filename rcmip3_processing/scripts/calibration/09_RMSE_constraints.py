import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# for RCMIP3: back to just GMST, air-sea CO2 flux

# run the priors (FRIDA-clim_priors.stmx) before this.

# from here onwards the output depends on the calibration version.

# taken from calibrate-FRIDA-climate

# Adapted from FaIR calibrate

# Exclude priors which don't closely match GMST and air-sea CO2 flux

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

calibration = os.getenv("CALIBRATION")

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))

weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5


#%%
df_obs = pd.read_csv(
    f'{indir}/rcmip_phase3_processed_constraining_data_{rcmip_version}.csv')

df_obs = df_obs.reset_index(drop=True)

df_temp = pd.read_csv("../../data/priors_output/priors_temperature.csv")

temp_hist = df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=2023)].drop(columns='Year').values
temp_hist_offset = temp_hist - np.average(temp_hist[:52, :], weights=weights, axis=0)

years_obs = [col for col in df_obs.columns if str(col).isdigit()]

gmst = df_obs.loc[df_obs["Variable"] == "Global Mean Surface Temperature (GMST)"][years_obs]

gmst_series = gmst[years_obs].iloc[0]
gmst_series.index = gmst_series.index.astype(int)

gmst = gmst_series.to_numpy()
time_temp = gmst_series.index.to_numpy()


df_flux = pd.read_csv("../../data/priors_output/priors_ocean_CO2_flux.csv")

flux_hist = df_flux.loc[(df_flux['Year']>=1781) & (df_flux['Year']<=2022)].drop(columns='Year').values
flux_hist_years = df_flux.loc[(df_flux['Year']>=1781) & (df_flux['Year']<=2022)]['Year'].values

flux_hist_for_rmse = df_flux.loc[(df_flux['Year']>=1960) & (df_flux['Year']<=2022)].drop(columns='Year').values

years_flux = [str(y) for y in 
              range(1960, 2023)]
flux_for_rmse = df_obs.loc[
    df_obs["Variable"] == "Carbon Flux to Oceans"
][years_flux].values[0,:]

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

fig, axs = plt.subplots(4, 2, figsize=(15, 15))

axs[0,0].fill_between(time_temp, np.percentile(temp_hist_offset, 84, axis=1), 
              np.percentile(temp_hist_offset, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,0].plot(time_temp, np.median(temp_hist_offset, axis=1), 
              color="#000000", label='Median')

axs[0,0].plot(time_temp, gmst, label='AR6 obs')

axs[0,0].legend(loc = 'upper left')
axs[0,0].set_ylabel('deg C')
axs[0,0].set_title(f'All priors: {samples}')



axs[0,1].fill_between(flux_hist_years, np.percentile(flux_hist, 84, axis=1), 
              np.percentile(flux_hist, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,1].plot(flux_hist_years, np.median(flux_hist, axis=1), 
              color="#000000", label='Median')

# axs[0,1].plot(flux_hist_years, flux)
axs[0,1].plot(years_flux, flux_for_rmse, label='GCB obs')

axs[0,1].legend()
axs[0,1].set_ylabel('GtC/yr')
axs[0,1].set_title(f'All priors: {samples}')



axs[1,0].fill_between(time_temp, np.percentile(temp_hist_offset[:, accept_temp], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,0].plot(time_temp, np.median(temp_hist_offset[:, accept_temp], axis=1), 
              color="#000000", label='Median')

axs[1,0].plot(time_temp, gmst, label='AR6 obs')

axs[1,0].legend()
axs[1,0].set_ylabel('deg C')
axs[1,0].set_title(f'Passing temp: {n_pass_temp}')



axs[1,1].fill_between(flux_hist_years, np.percentile(flux_hist[:, accept_temp], 84, axis=1), 
              np.percentile(flux_hist[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,1].plot(flux_hist_years, np.median(flux_hist[:, accept_temp], axis=1), 
              color="#000000", label='Median')

# axs[1,1].plot(flux_hist_years, flux)
axs[1,1].plot(years_flux, flux_for_rmse, label='GCB obs')

axs[1,1].legend()
axs[1,1].set_ylabel('GtC/yr')
axs[1,1].set_title(f'Passing temp: {n_pass_temp}')




axs[2,0].fill_between(time_temp, np.percentile(temp_hist_offset[:, accept_flux], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,0].plot(time_temp, np.median(temp_hist_offset[:, accept_flux], axis=1), 
              color="#000000", label='Median')

axs[2,0].plot(time_temp, gmst, label='AR6 obs')

axs[2,0].legend()
axs[2,0].set_ylabel('deg C')
axs[2,0].set_title(f'Passing flux: {n_pass_flux}')



axs[2,1].fill_between(flux_hist_years, np.percentile(flux_hist[:, accept_flux], 84, axis=1), 
              np.percentile(flux_hist[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,1].plot(flux_hist_years, np.median(flux_hist[:, accept_flux], axis=1), 
              color="#000000", label='Median')

# axs[2,1].plot(flux_hist_years, flux)
axs[2,1].plot(years_flux, flux_for_rmse, label='GCB obs')

axs[2,1].legend()
axs[2,1].set_ylabel('GtC/yr')
axs[2,1].set_title(f'Passing flux: {n_pass_flux}')





axs[3,0].fill_between(time_temp, np.percentile(temp_hist_offset[:, accept_both], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,0].plot(time_temp, np.median(temp_hist_offset[:, accept_both], axis=1), 
              color="#000000", label='Median')

axs[3,0].plot(time_temp, gmst, label='AR6 obs')

axs[3,0].legend()
axs[3,0].set_ylabel('deg C')
axs[3,0].set_title(f'Passing both: {n_pass_both}')



axs[3,1].fill_between(flux_hist_years, np.percentile(flux_hist[:, accept_both], 84, axis=1), 
              np.percentile(flux_hist[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,1].plot(flux_hist_years, np.median(flux_hist[:, accept_both], axis=1), 
              color="#000000", label='Median')

# axs[3,1].plot(flux_hist_years, flux)
axs[3,1].plot(years_flux, flux_for_rmse, label='GCB obs')

axs[3,1].legend()
axs[3,1].set_ylabel('GtC/yr')
axs[3,1].set_title(f'Passing both: {n_pass_both}')

for i in np.arange(4):
    for j in np.arange(2):
        axs[i,j].set_xlim([1850, 2022])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=10, hspace=None)

plt.tight_layout()

os.makedirs("../../calibration/plots", exist_ok=True)

plt.savefig(
    "../../calibration/plots/rmse_constrained.png"
)

#%%
np.savetxt(
    "../../data/constraining/runids_rmse_pass.csv",
    valid_both.astype(int),
    fmt="%d",
)

