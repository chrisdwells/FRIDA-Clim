import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

start = 1750
end = 2500
n_years = end - start + 1

pi_years = [1850, 1900]
pi_idxs = [idx - start for idx in pi_years]

pd_years = [1995, 2014]
pd_idxs = [idx - start for idx in pi_years]


time = np.arange(start, end+1, 1)

load_dotenv()

output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

varlist = [
       "Energy Balance Model.Land & Ocean Surface Temperature[1]",
       # "CO2 Forcing.Atmospheric CO2 Concentration[1]",
       # "Forcing.Total Effective Radiative Forcing[1]",
       # "Ocean.Air sea co2 flux[1]",
       # "Energy Balance Model.ocean heat content change[1]",
       "Terrestrial Carbon Balance.Terrestrial carbon balance[1]",
       # "Emissions.CO2 Emissions from Food and Land Use[1]",
       # "Sea Level.Total global sea level anomaly[1]",
       # "Terrestrial Carbon Balance.Terrestrial net primary production[1]",
       # "Grass.grassland net primary production[1]",
       # "Forest.young forest net primary production[1]",
       # "Forest.mature forest net primary production[1]",
       # "Terrestrial Carbon Balance.Total soil carbon[1]",
       # "Terrestrial Carbon Balance.Total land carbon[1]",
       # "Ocean.Cold surface ocean pH[1]",
       # "Ocean.Warm surface ocean pH[1]",
       # "Module.Land Carbon Sink[1]",
       ]

offset_vars = {
       "Energy Balance Model.Land & Ocean Surface Temperature[1]":pi_idxs,
       "Energy Balance Model.ocean heat content change[1]":pd_idxs,
    }

calc_vars = {
    "Module.Land Carbon Sink[1]":'TCB + CO2 FLU'
    }

obs_data = {}

# IGCC as used for calibration
df_temp_obs = pd.read_csv("../data/external/forcing/annual_averages.csv")
gmst = df_temp_obs["gmst"] - np.mean(df_temp_obs["gmst"].loc[(df_temp_obs['time'] > pi_years[0]) 
                               & (df_temp_obs['time'] < pi_years[1])].values)

obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"] = [
    df_temp_obs['time'], gmst
    ]

# IGCC
df_co2_obs = pd.read_csv("../data/external/forcing/ghg_concentrations.csv")

obs_data["CO2 Forcing.Atmospheric CO2 Concentration[1]"] = [
    df_co2_obs['time'], df_co2_obs['CO2']]

# IGCC
df_erf_obs = pd.read_csv("../data/external/forcing/ERF_best_aggregates_1750-2024.csv")
obs_data["Forcing.Total Effective Radiative Forcing[1]"] = [
    df_erf_obs['time'], df_erf_obs['total']]


# IGCC
df_slr_obs = pd.read_csv("../data/external/forcing/IGCC_GMSL_ensemble.csv")
obs_data["Sea Level.Total global sea level anomaly[1]"] = [
    df_slr_obs['time'], df_slr_obs['mean']/1000] # mm to m


# GCB
gcb = pd.read_csv("../data/external/gcp_v2023_co2_1750-2022.csv")
gcb = gcb.loc[gcb['Year'] <= 2022]

obs_data["Module.Land Carbon Sink[1]"] = [
    gcb['Year'], gcb['land sink']]
obs_data["Terrestrial Carbon Balance.Terrestrial carbon balance[1]"] = [
    gcb['Year'], gcb['terrestrial carbon balance']]


df_ocean = pd.read_csv("../data/external/GCB_historical_budget.csv")

obs_data["Ocean.Air sea co2 flux[1]"] = [
    df_ocean['Year'], df_ocean['ocean sink']]

ssps = {
"ssp119":"AR6-SSP1-1.9",
"ssp126":"AR6-SSP1-2.6",
"ssp245":"AR6-SSP2-4.5",
"ssp370":"AR6-SSP3-7.0",
"ssp434":"AR6-SSP4-3.4",
"ssp460":"AR6-SSP4-6.0",
"ssp534-over":"AR6-SSP5-3.4-OS",
"ssp585":"AR6-SSP5-8.5",
    }
with open('../data/external/misc/colors_pd.pkl', 'rb') as handle:
    colors_pd = pickle.load(handle)

#%%
for var in varlist:
    
    varname = var.split(".")[1].split("[1]")[0]
    
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax = ax.ravel()
    
    for s_i, scen in enumerate(ssps.keys()):
        
        color=colors_pd.loc[
                colors_pd['name'] == ssps[scen]]['color'].values[0]

        ssp_in = pd.read_csv(f'../data/ssps_output/{scen}_output.csv')
        
        var_data = np.full((n_years, output_ensemble_size), np.nan)

        if var in calc_vars.keys():
            if var == "Module.Land Carbon Sink[1]":
                tcb_data = np.full((n_years, output_ensemble_size), np.nan)
                co2_flu_data = np.full((n_years, output_ensemble_size), np.nan)
                for i in np.arange(output_ensemble_size):
                    var_data[:,i] = ssp_in[f'="Run {i+1}: Terrestrial Carbon Balance.Terrestrial carbon balance[1]"'
                               ] + ssp_in[f'="Run {i+1}: Emissions.CO2 Emissions from Food and Land Use[1]"']/3670 # MtCO2 to GtC
            else:
                print('Need to make definition')
        else:
            for i in np.arange(output_ensemble_size):
                var_data[:,i] = ssp_in[f'="Run {i+1}: {var}"']
                
        offset_text = ''
        if var in offset_vars.keys():
            idxs = offset_vars[var]
            offset_text = f' (offset {idxs[0]+start}-{idxs[1]+start})'
            var_data = var_data - np.mean(var_data[idxs[0]:idxs[1],:], axis=0)
            
            ax[s_i].axhline(0, color='grey', linestyle = '--')

            
        ax[s_i].plot(time, np.median(var_data, axis=1), color=color)
        
        ax[s_i].fill_between(time, 
                         np.percentile(var_data, 16, axis=1), 
                         np.percentile(var_data, 84, axis=1), 
                         color=color, linewidth=0, alpha=0.2)

        ax[s_i].fill_between(time, 
                         np.percentile(var_data, 5, axis=1), 
                         np.percentile(var_data, 95, axis=1), 
                         color=color, linewidth=0, alpha=0.2)
        

        if var in obs_data.keys():
            ax[s_i].plot(obs_data[var][0], obs_data[var][1], color='black')
    
        ax[s_i].set_title(f'{scen}')
    
    plt.suptitle(f'{varname}{offset_text}, {output_ensemble_size} members')
    plt.tight_layout()
    plt.savefig(
        f"../plots/ssps/{varname}.png"
    )

#%%

vars_plot = {
    "Energy Balance Model.Land & Ocean Surface Temperature[1]":['K',-1, 10],
    "Terrestrial Carbon Balance.Terrestrial carbon balance[1]":['GtC/year', -2, 4],
    }

scens_plot = ['ssp119', 'ssp245', 'ssp534-over', 'ssp585']

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
ax = ax.ravel()

title_text = ''
c = -1
for var in vars_plot.keys():
    varname = var.split(".")[1].split("[1]")[0]

    for scen in scens_plot:
        c += 1
        
        color=colors_pd.loc[
                colors_pd['name'] == ssps[scen]]['color'].values[0]
    
        ssp_in = pd.read_csv(f'../data/ssps_output/{scen}_output.csv')
        
        var_data = np.full((n_years, output_ensemble_size), np.nan)
    
        if var in calc_vars.keys():
            if var == "Module.Land Carbon Sink[1]":
                tcb_data = np.full((n_years, output_ensemble_size), np.nan)
                co2_flu_data = np.full((n_years, output_ensemble_size), np.nan)
                for i in np.arange(output_ensemble_size):
                    var_data[:,i] = ssp_in[f'="Run {i+1}: Terrestrial Carbon Balance.Terrestrial carbon balance[1]"'
                               ] + ssp_in[f'="Run {i+1}: Emissions.CO2 Emissions from Food and Land Use[1]"']/3670 # MtCO2 to GtC
            else:
                print('Need to make definition')
        else:
            for i in np.arange(output_ensemble_size):
                var_data[:,i] = ssp_in[f'="Run {i+1}: {var}"']
                
        offset_text = ''
        if var in offset_vars.keys():
            idxs = offset_vars[var]
            offset_text = f' (offset {idxs[0]+start}-{idxs[1]+start})'
            var_data = var_data - np.mean(var_data[idxs[0]:idxs[1],:], axis=0)
            
        ax[c].axhline(0, color='grey', linestyle = '--')
        
            
        ax[c].plot(time, np.median(var_data, axis=1), color=color)
        
        ax[c].fill_between(time, 
                         np.percentile(var_data, 16, axis=1), 
                         np.percentile(var_data, 84, axis=1), 
                         color=color, linewidth=0, alpha=0.2)
    
        ax[c].fill_between(time, 
                         np.percentile(var_data, 5, axis=1), 
                         np.percentile(var_data, 95, axis=1), 
                         color=color, linewidth=0, alpha=0.2)
        
    
        if var in obs_data.keys():
            ax[c].plot(obs_data[var][0], obs_data[var][1], color='black')
    
        ax[c].set_title(f'{scen}')
        if c % 4 == 0:
            ax[c].set_ylabel(f'{vars_plot[var][0]}')
            
        ax[c].set_xlim([start, end])
        ax[c].set_ylim([vars_plot[var][1], vars_plot[var][2]])


    title_text = title_text + f'; {varname}{offset_text}'


leg_color = colors_pd.loc[
        colors_pd['name'] == ssps['ssp119']]['color'].values[0]

handles = []

handles.append(Line2D([0], [0], label='Observations', color='black'))
handles.append(Line2D([0], [0], label='Median', color=leg_color))

handles.append(mpatches.Patch(facecolor=leg_color, edgecolor=leg_color, linewidth=0, alpha=0.2, label='5-95 percentile'))
handles.append(mpatches.Patch(facecolor=leg_color, edgecolor=leg_color, linewidth=0, alpha=0.4, label='16-84 percentile'))

ax[0].legend(handles=handles)

plt.suptitle(f'{title_text[1:]}, {output_ensemble_size} members')
plt.tight_layout()
plt.savefig(
    "../plots/ssps/fig3_paper.png"
)
