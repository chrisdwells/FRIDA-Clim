import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

calibration = 'RCMIP' # we just compare the FRIDA-Clim RCMIP calibration here
output_ensemble_size = 100

start = 1750
end = 2500
n_years = end - start + 1
time = np.arange(start, end+1, 1)


with open('../../data/external/misc/colors_pd.pkl', 'rb') as handle:
    colors_pd = pickle.load(handle)

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

def loaddata(df, n_years, members, varname, offset=False):
    var_data = np.full((n_years, members), np.nan)
    for i in np.arange(members):
        var_data[:,i] = df[f'="Run {i+1}: {varname}"'][:n_years]
    if offset == True:
        var_data = var_data - var_data[0,:]
    return var_data

def run_fair_var_co2(co2_conc, scenario):
    f = FAIR()
    
    f.define_time(1750, 2500, 1)
    
    f.define_scenarios([scenario])
    
    configs = ['0.05', '0.17', '0.5', '0.83', '0.95']
    f.define_configs(configs)
    
    species, properties = read_properties()
    properties['CO2']['input_mode'] = 'concentration'
    
    f.define_species(species, properties)
    
    f.allocate()
    f.fill_species_configs()
    f.fill_from_rcmip()
    
    for config in configs:
        f.concentration[:, 0, f.configs.index(config), f.species.index("CO2")] = co2_conc[config]

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    
    capacities = [4.22335014, 16.5073541, 86.1841127]
    kappas = [1.31180598, 2.61194068, 0.92986733]
    epsilon = 1.29020599
    fill(f.climate_configs['ocean_heat_capacity'], capacities)
    fill(f.climate_configs['ocean_heat_transfer'], kappas)
    fill(f.climate_configs['deep_ocean_efficacy'], epsilon)
    
    f.run()
    
    return f.temperature.sel(scenario = scenario, layer=0)
    

scens_plot = [
    "ssp119",
    "ssp245",
    "ssp534-over",
    "ssp585",
]

df = {}
df['vary_co2'] = {}
df['vary_ebm'] = {}

for s_i, scen in enumerate(scens_plot):
    df['vary_co2'][scen] = {}
    df['vary_ebm'][scen] = {}
    
    # get the CO2 conc from the full FaIR, FRIDA-Clim runs, put through FaIR default
    ssp_rcmip_frida = pd.read_csv(f'../../{calibration}/data/ssps_output/{scen}_output.csv')
    co2_conc_frida_full = loaddata(ssp_rcmip_frida, n_years, output_ensemble_size, "CO2 Forcing.Atmospheric CO2 Concentration[1]", 
                      offset=False)
    co2_conc_frida = {}
    co2_conc_frida['0.05'], co2_conc_frida['0.17'], co2_conc_frida['0.5'
       ], co2_conc_frida['0.83'], co2_conc_frida['0.95'] = np.percentile(co2_conc_frida_full, 
           [5, 17, 50, 83, 95], axis=1) 
    gmst_frida = run_fair_var_co2(co2_conc_frida, scen)
    df['vary_co2'][scen]['frida'] = gmst_frida

    co2_conc_fair = pd.read_csv(f'../../data/external/fair_input/RCMIP/co2_concentration_{scen}.csv')
    gmst_fair = run_fair_var_co2(co2_conc_fair, scen)
    df['vary_co2'][scen]['fair'] = gmst_fair

    # get the temperatures when running RFMIP ERFs through FaIR (see prior script)
    # and FRIDA-Clim (see _RFMIP model in RCMIP calibration, and frida_clim_processing for script)
    ssp_rfmip_frida = pd.read_csv(f'../../{calibration}/data/ssps_output/RFMIP/{scen}_RFMIP.csv')
    gmst_frida_full = loaddata(ssp_rfmip_frida, n_years, output_ensemble_size, "Energy Balance Model.Land & Ocean Surface Temperature[1]", 
                      offset=True)
    df['vary_ebm'][scen]['frida'] = gmst_frida_full

    gmst_fair_rfmip = pd.read_csv(f'../../data/external/fair_input/RFMIP/temperature_{scen}.csv')
    df['vary_ebm'][scen]['fair'] = gmst_fair_rfmip

#%%

pi_years = [1850, 1900]
obs_data = {}

df_temp_obs = pd.read_csv("../../data/external/forcing/annual_averages.csv")
gmst = df_temp_obs["gmst"] - np.mean(df_temp_obs["gmst"].loc[(df_temp_obs['time'] > pi_years[0]) 
                               & (df_temp_obs['time'] < pi_years[1])].values)

obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"] = [
    df_temp_obs['time'], gmst
    ]

fig, ax = plt.subplots(2, 4, figsize=(12, 9))

for s_i, scen in enumerate(scens_plot):
    color=colors_pd.loc[
            colors_pd['name'] == ssps[scen]]['color'].values[0]


    ax[0,s_i].axhline(0, color='grey', linestyle = '--')
    ax[0,s_i].set_ylim([-1, 13])
    ax[0,s_i].set_ylabel('K')

    ax[0, s_i].plot(obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"][
        0], obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"][1], color='black')

    
    ax[0,s_i].plot(time, df['vary_co2'][scen]['frida'].sel(config='0.5').values, color=color)
    ax[0,s_i].fill_between(time, df['vary_co2'][scen]['frida'].sel(config='0.05').values,
                           df['vary_co2'][scen]['frida'].sel(config='0.95').values, color=color, 
                           linewidth=0, alpha=0.2)
    
    
    ax[0,s_i].plot(time, df['vary_co2'][scen]['fair'].sel(config='0.5').values, color='grey')
    ax[0,s_i].fill_between(time, df['vary_co2'][scen]['fair'].sel(config='0.05').values,
                           df['vary_co2'][scen]['fair'].sel(config='0.95').values, color='grey', 
                           linewidth=0, alpha=0.2, hatch='//')
    


    ax[1,s_i].axhline(0, color='grey', linestyle = '--')
    ax[1,s_i].set_ylim([-1, 13])
    ax[1,s_i].set_ylabel('K')
    ax[1, s_i].plot(obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"][
        0], obs_data["Energy Balance Model.Land & Ocean Surface Temperature[1]"][1], color='black')

    
    ax[1,s_i].plot(time, np.percentile(df['vary_ebm'][scen]['frida'], 50, axis=1), color=color)
    ax[1,s_i].fill_between(time, np.percentile(df['vary_ebm'][scen]['frida'], 5, axis=1),
                           np.percentile(df['vary_ebm'][scen]['frida'], 95, axis=1), 
                           color=color, linewidth=0, alpha=0.2)
    
    ax[1,s_i].plot(time, df['vary_ebm'][scen]['fair']['0.5'], color='grey')
    ax[1,s_i].fill_between(time, df['vary_ebm'][scen]['fair']['0.05'],
                           df['vary_ebm'][scen]['fair']['0.95'], 
                           color='grey', linewidth=0, alpha=0.2, hatch='//')
        

leg_color = colors_pd.loc[
        colors_pd['name'] == ssps['ssp119']]['color'].values[0]

handles = []
handles.append(Line2D([0], [0], label='Observations', color='black'))

handles.append(mpatches.Patch(facecolor=leg_color, edgecolor=leg_color, linewidth=0, alpha=0.2, label='5-95 percentile'))
handles.append(Line2D([0], [0], label='Median', color=leg_color))

handles.append(mpatches.Patch(facecolor=leg_color, edgecolor=leg_color, linewidth=0, alpha=0.3, label='FRIDA-Clim'))
handles.append(mpatches.Patch(facecolor='grey', edgecolor='grey', linewidth=0, alpha=0.3, hatch='//', label='FaIR'))
ax[0, 0].legend(handles=handles, ncol=1, fontsize=10)


handles = []
for s_i, scen in enumerate(scens_plot):
    color=colors_pd.loc[
            colors_pd['name'] == ssps[scen]]['color'].values[0]
    handles.append(Line2D([0], [0], label=scen, color=color))

ax[1, 0].legend(handles=handles, ncol=1, fontsize=10)
    
plt.suptitle('GMST response in FRIDA-Clim and FaIR under varying CO2 concentration in same EBM (top), identical forcings in separate EBMs (bottom)')
plt.tight_layout()
plt.savefig(
    f"../../{calibration}/plots/ssps/figS6_paper.png"
)
    
    
    
