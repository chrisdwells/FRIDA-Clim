import pickle 
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr


datadir = 'data'
figdir = 'plots'

with open(f'{datadir}/scenario_info.pkl', 'rb') as infile:
   file_data = pickle.load(infile)
   
with open(f'{datadir}/scenario_info_n2o_nox.pkl', 'rb') as infile:
   file_data_n2o_nox = pickle.load(infile)
   
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

f = FAIR()

f.define_time(1750, 2015, 1)

scenarios = ['ssp119']
f.define_scenarios(scenarios)

configs = ['test']
f.define_configs(configs)

species, properties = read_properties()
f.define_species(species, properties)

f.allocate()
f.fill_species_configs()
f.fill_from_rcmip()

full_fair_ems = np.zeros((350, 1703, 1, 64))

full_fair_ems[265:,...] = file_data['Emissions']
for s_i in np.arange(1703):
    full_fair_ems[:265,s_i,...] = f.emissions[:,0,:,:]

#%%

f = FAIR()

f.define_time(1750, 2100, 1)

scenarios = ['ssp119']
f.define_scenarios(scenarios)

configs = ['test']
f.define_configs(configs)

species, properties = read_properties()
f.define_species(species, properties)

f.allocate()
f.fill_species_configs()
f.fill_from_rcmip()

full_fair_forcing = np.ones((351, 1703, 1, 64)) * np.nan
for s_i in np.arange(1703):
    full_fair_forcing[:,s_i,:,54] = f.forcing[:,0,:,54]
    full_fair_forcing[:,s_i,:,55] = f.forcing[:,0,:,55]


#%%
scenarios = []
for scen in file_data['Scenarios']:
    scenarios.append(scen)
    

f_original = FAIR()

f_original.define_time(1750, 2100, 1)

f_original.define_scenarios(scenarios)

configs = ['test']
f_original.define_configs(configs)

species, properties = read_properties()
f_original.define_species(species, properties)

f_original.allocate()
f_original.fill_species_configs()

fill(f_original.forcing, full_fair_forcing)

fill(f_original.emissions, full_fair_ems)

initialise(f_original.concentration, f_original.species_configs['baseline_concentration'])
initialise(f_original.forcing, 0)
initialise(f_original.temperature, 0)
initialise(f_original.cumulative_emissions, 0)
initialise(f_original.airborne_emissions, 0)

capacities = [4.22335014, 16.5073541, 86.1841127]
kappas = [1.31180598, 2.61194068, 0.92986733]
epsilon = 1.29020599
fill(f_original.climate_configs['ocean_heat_capacity'], capacities)
fill(f_original.climate_configs['ocean_heat_transfer'], kappas)
fill(f_original.climate_configs['deep_ocean_efficacy'], epsilon)

f_original.run()
#%%

potential_links = {
    'BC Snow':['CO2 AFOLU Emissions', 'Sulfur Emissions'],
    'VOC Emissions':['CH4 Emissions'],
    # 'NOx Emissions':['N2O Emissions'],
    'CO Emissions':['CH4 Emissions'],
    'NOx non-AFOLU Emissions':['N2O non-AFOLU Emissions'],
    'NOx AFOLU Emissions':['Sulfur Emissions for NOx'],
    }

nt_bounds = 351
nt_points = 350

interpolator = scipy.interpolate.interp1d(
    np.arange(nt_bounds),
    f_original.forcing[:,:,:,f_original.species.index("Aerosol-cloud interactions")],
    axis=0,
)
aci_interp = interpolator(np.arange(nt_points))

interpolator = scipy.interpolate.interp1d(
    np.arange(nt_bounds),
    f_original.forcing[:,:,:,f_original.species.index("Ozone")],
    axis=0,
)
ozone_forcing_interp = interpolator(np.arange(nt_points))

interpolator = scipy.interpolate.interp1d(
    np.arange(nt_bounds),
    f_original.concentration[:,:,:,f_original.species.index("CH4")],
    axis=0,
)
ch4_conc_interp = interpolator(np.arange(nt_points))

interpolator = scipy.interpolate.interp1d(
    np.arange(nt_bounds),
    f_original.concentration[:,:,:,f_original.species.index("N2O")],
    axis=0,
)
n2o_conc_interp = interpolator(np.arange(nt_points))

interpolator = scipy.interpolate.interp1d(
    np.arange(nt_bounds),
    f_original.concentration[:,:,:,f_original.species.index("Equivalent effective stratospheric chlorine")],
    axis=0,
)
eesc_conc_interp = interpolator(np.arange(nt_points))

vars_calc_future = {
    'BC Snow':f_original.forcing[265:,:,:,f_original.species.index("Light absorbing particles on snow and ice")],

    'VOC Emissions':f_original.emissions[265:,:,:,f_original.species.index('VOC')].values - f_original.emissions[0,:,:,f_original.species.index('VOC')].values,
    'NOx Emissions':f_original.emissions[265:,:,:,f_original.species.index('NOx')].values - f_original.emissions[0,:,:,f_original.species.index('NOx')].values,
    'CO Emissions':f_original.emissions[265:,:,:,f_original.species.index('CO')].values - f_original.emissions[0,:,:,f_original.species.index('CO')].values,
    
    'CO2 AFOLU Emissions':f_original.emissions[265:,:,:,f_original.species.index("CO2 AFOLU")] - f_original.emissions[0,:,:,f_original.species.index("CO2 AFOLU")],
    'Sulfur Emissions':f_original.emissions[265:,:,:,f_original.species.index("Sulfur")] - f_original.emissions[0,:,:,f_original.species.index("Sulfur")],
    'CH4 Emissions':f_original.emissions[265:,:,:,f_original.species.index("CH4")] - f_original.emissions[0,:,:,f_original.species.index("CH4")],
    'N2O Emissions':f_original.emissions[265:,:,:,f_original.species.index("N2O")] - f_original.emissions[0,:,:,f_original.species.index("N2O")],
    
    'NOx non-AFOLU Emissions':(file_data_n2o_nox['NOx|non-AFOLU'][265:-1,:] - file_data_n2o_nox['NOx|non-AFOLU'][0,:])[:,:,None],
    'NOx AFOLU Emissions':(file_data_n2o_nox['NOx|AFOLU'][265:-1,:] - file_data_n2o_nox['NOx|AFOLU'][0,:])[:,:,None],
    'N2O non-AFOLU Emissions':(file_data_n2o_nox['N2O|non-AFOLU'][265:-1,:] - file_data_n2o_nox['N2O|non-AFOLU'][0,:])[:,:,None],
    'Sulfur Emissions for NOx':(file_data_n2o_nox['Sulfur'][265:-1,:] - file_data_n2o_nox['Sulfur'][0,:])[:,:,None],
    }

vars_calc_hist = {
    'BC Snow':f_original.forcing[:265,0:1,:,f_original.species.index("Light absorbing particles on snow and ice")],

    'VOC Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("VOC")] - f_original.emissions[0,0:1,:,f_original.species.index("VOC")],
    'NOx Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("NOx")] - f_original.emissions[0,0:1,:,f_original.species.index("NOx")],
    'CO Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("CO")] - f_original.emissions[0,0:1,:,f_original.species.index("CO")],

    'CO2 AFOLU Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("CO2 AFOLU")] - f_original.emissions[0,0:1,:,f_original.species.index("CO2 AFOLU")],
    'Sulfur Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("Sulfur")] - f_original.emissions[0,0:1,:,f_original.species.index("Sulfur")],
    'CH4 Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("CH4")] - f_original.emissions[0,0:1,:,f_original.species.index("CH4")],
    'N2O Emissions':f_original.emissions[:265,0:1,:,f_original.species.index("N2O")] - f_original.emissions[0,0:1,:,f_original.species.index("N2O")],

    'NOx non-AFOLU Emissions':xr.DataArray((file_data_n2o_nox['NOx|non-AFOLU'][:265,0:1] - file_data_n2o_nox['NOx|non-AFOLU'][0,0])[:,:,None]),
    'NOx AFOLU Emissions':xr.DataArray((file_data_n2o_nox['NOx|AFOLU'][:265,0:1] - file_data_n2o_nox['NOx|AFOLU'][0,0])[:,:,None]),
    'N2O non-AFOLU Emissions':xr.DataArray((file_data_n2o_nox['N2O|non-AFOLU'][:265,0:1] - file_data_n2o_nox['N2O|non-AFOLU'][0,0])[:,:,None]),
    'Sulfur Emissions for NOx':xr.DataArray((file_data_n2o_nox['Sulfur'][:265,0:1] - file_data_n2o_nox['Sulfur'][0,0])[:,:,None]),
    }


#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np

regr = LinearRegression(fit_intercept=False)

from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


nt_future_points = 85
nt_future_bounds = 86
nt_hist_points = 265
nt_hist_bounds = 266

regr_data_for_plot = {}

for targ in potential_links.keys():
    regr_data_for_plot[targ] = {}
    
    if "NOx" in targ:
        n_scens = len(file_data_n2o_nox['Scenarios']) 
    else:
        n_scens = len(file_data['Scenarios'])

    

    preds = powerset(potential_links[targ])
    preds_list = []
    for pred in preds:
        pred_list = []
        for p_i in np.arange(len(pred)):
            pred_list.append(pred[p_i])
        if len(pred_list) > 0:
            preds_list.append(pred_list)
        
        
    regr_data_for_plot[targ]['Preds'] = preds_list

    coef_list = []
    r2_list = []
    int_list = []
    
    for ni in np.arange(len(preds_list)):
        pred = preds_list[ni]
        pred_string= ''
        for p in pred:
            pred_string = pred_string + ' + ' + p
        
        regr_data_for_plot[targ][pred_string] = {}
        pred_data = np.zeros((nt_future_points, n_scens, len(pred)))
        for pred_i, pred_component in enumerate(pred):
            pred_data[:,:,pred_i] = vars_calc_future[pred_component][:,:,0]
            pred_array = pred_data.reshape(-1, pred_data.shape[-1])
            
        
        if vars_calc_future[targ].shape[0] == nt_future_points:
            targ_data = vars_calc_future[targ]
            targ_array = targ_data.flatten()
        else:
            interpolator = scipy.interpolate.interp1d(
                np.arange(nt_future_bounds),
                vars_calc_future[targ],
                axis=0,
            )
            targ_data = interpolator(np.arange(nt_future_points))
            targ_array = targ_data.flatten()

        regr.fit(pred_array, targ_array)
        
        coef_list.append(regr.coef_)
        
        pred_for_error = regr.predict(pred_array)
        
        r2_list.append(r2_score(targ_array, pred_for_error))
        
        int_list.append(regr.intercept_)
        
        regr_data_for_plot[targ][pred_string]['Future_targ_data'] = targ_data
        regr_data_for_plot[targ][pred_string]['Future_pred_data'] = pred_data

            
    regr_data_for_plot[targ]['Future_coefs'] = coef_list
    regr_data_for_plot[targ]['Future_r2'] = r2_list
    regr_data_for_plot[targ]['Future_int'] = int_list

    
    coef_list = []
    r2_list = []
    int_list = []
    
    for ni in np.arange(len(preds_list)):
        pred = preds_list[ni]
        pred_string= ''
        for p in pred:
            pred_string = pred_string + ' + ' + p
            
        pred_array = np.zeros((nt_hist_points, len(pred)))
        for pred_i, pred_component in enumerate(pred):
            pred_array[:,pred_i] = np.asarray(vars_calc_hist[pred_component]).flatten()
            
        
        if vars_calc_hist[targ].shape[0] == nt_hist_points:
            targ_data = vars_calc_hist[targ]
            targ_array = targ_data.values.flatten()
        else:
            interpolator = scipy.interpolate.interp1d(
                np.arange(nt_hist_bounds),
                vars_calc_hist[targ],
                axis=0,
            )
            targ_data = interpolator(np.arange(nt_hist_points))
            targ_array = targ_data.flatten()

        
        regr.fit(pred_array, targ_array)
        
        coef_list.append(regr.coef_)
        
        pred_for_error = regr.predict(pred_array)
        
        r2_list.append(r2_score(targ_array, pred_for_error))
            
        int_list.append(regr.intercept_)

        regr_data_for_plot[targ][pred_string]['Historical_targ_data'] = targ_data
        regr_data_for_plot[targ][pred_string]['Historical_pred_data'] = pred_array

    regr_data_for_plot[targ]['Hist_coefs'] = coef_list
    regr_data_for_plot[targ]['Hist_r2'] = r2_list
    regr_data_for_plot[targ]['Hist_int'] = int_list


#%%
idx_dict = {}
for targ in potential_links.keys():
    preds_list = regr_data_for_plot[targ]['Preds']
    # fig, axs = plt.subplots(4, 1, figsize=(1.8*len(preds_list), 12))
    
    perc_dif = np.zeros(len(preds_list))
    for p_i, pred in enumerate(preds_list):
        perc_dif[p_i] = np.mean(100*(np.abs(regr_data_for_plot[targ]['Future_coefs'][p_i] - regr_data_for_plot[targ]['Hist_coefs'][p_i])/np.abs(np.abs(regr_data_for_plot[targ]['Hist_coefs'][p_i]))))

    sorted_idxs = np.argsort(perc_dif)
    idx_dict[targ] = sorted_idxs
#%%

import matplotlib as mpl
import copy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({'mathtext.default':  'regular' })

# predictors, and target name in FaIR
regr_data_for_plot_actual = {
    'BC Snow':[[['CO2 AFOLU Emissions', 'Sulfur Emissions']], 'nan'],
    'VOC Emissions':[[['CH4 Emissions']], 'VOC'],
    # 'NOx Emissions':[['N2O Emissions']],
    'CO Emissions':[[['CH4 Emissions']], 'CO'],
    'NOx non-AFOLU Emissions':[[['N2O non-AFOLU Emissions']], 'NOx'],
    'NOx AFOLU Emissions':[[['Sulfur Emissions for NOx']], 'NOx'],
    }

time = 1750 + np.arange(nt_hist_points + nt_future_points)
time_gmst = 1750 + np.arange(nt_hist_points + nt_future_points+1)


f = FAIR()

f.define_time(1750, 2100, 1)

scenarios = ['ssp245']
f.define_scenarios(scenarios)

configs = ['test']
f.define_configs(configs)

species, properties = read_properties()
f.define_species(species, properties)

f.allocate()
f.fill_species_configs()
f.fill_from_rcmip()

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

gmst_ssp245 = copy.deepcopy(f.temperature.sel(
    scenario="ssp245",
    config="test",
    layer=0
))

fig, axs = plt.subplots(5, 2, figsize=(10, 15))
targ_i = -1
for targ in potential_links.keys():
    preds_list = regr_data_for_plot[targ]['Preds']

    if "NOx" in targ:
        n_scens = len(file_data_n2o_nox['Scenarios']) 
    else:
        n_scens = len(file_data['Scenarios'])

    idxs = idx_dict[targ]

    for pred_name in preds_list:
        if pred_name in regr_data_for_plot_actual[targ][0]:
            targ_i += 1
            
            p_i = preds_list.index(pred_name)
    
            ax = axs[targ_i, 0]
            ax.axhline(0, linestyle = '--', color = 'grey')

            pred = pred_name
            pred_string= ''
            for p in pred:
                pred_string = pred_string + ' + ' + p
            
            
            pred_name_lines = ''
            for pred_single in pred:
                pred_name_lines = pred_name_lines + ' ' + pred_single + ','
            ax.set_title(f'Target: {targ}; Predictor:{pred_name_lines[:-1]}', loc="left")
            
            coefs_hist = regr_data_for_plot[targ]['Hist_coefs'][p_i]
            intercept_hist = regr_data_for_plot[targ]['Hist_int'][p_i]
            
            coefs_future = regr_data_for_plot[targ]['Future_coefs'][p_i]
            intercept_future = regr_data_for_plot[targ]['Future_int'][p_i]
            
            targ_array = np.zeros((time.shape[0], n_scens))
            hist_model_array = np.zeros((time.shape[0], n_scens))
    
            for scen_i in np.arange(n_scens):
                targ_timeseries = np.concatenate((regr_data_for_plot[targ][pred_string]['Historical_targ_data'][:,0,0], regr_data_for_plot[targ][pred_string]['Future_targ_data'][:,scen_i,0]))
                targ_array[:,scen_i] = targ_timeseries
                pred_timeseries = np.zeros((time.shape[0], len(pred)))
                
                hist_model = np.zeros(time.shape[0])
                hist_model.fill(intercept_hist)
                for pred_component_i, pred_component in enumerate(pred):
                    pred_timeseries[:,pred_component_i] = np.concatenate((regr_data_for_plot[targ][pred_string]['Historical_pred_data'][:,pred_component_i], regr_data_for_plot[targ][pred_string]['Future_pred_data'][:,scen_i,pred_component_i]))
                    hist_model += coefs_hist[pred_component_i]*pred_timeseries[:,pred_component_i]
                
    
                hist_model_array[:,scen_i] = hist_model
    
            targ_50, targ_90, targ_95, targ_10, targ_5 = np.percentile(targ_array, [50, 90, 95, 10, 5], axis=1)
            hist_model_50, hist_model_90, hist_model_95, hist_model_10, hist_model_5 = np.percentile(hist_model_array, [50, 90, 95, 10, 5], axis=1)
    
            if targ in ['BC Snow', 'NOx non-AFOLU Emissions']:
                baseline = 0
            else:
                baseline = float(f_original.emissions[0,0:1,:,f_original.species.index(regr_data_for_plot_actual[targ][1])])

            ax.plot(time, baseline+targ_50, color = 'C0', alpha=1)
            ax.fill_between(time, baseline+targ_10, baseline+targ_90, color = 'C0', linewidth=0, alpha=0.5)
            ax.fill_between(time, baseline+targ_5, baseline+targ_95, color = 'C0', linewidth=0, alpha=0.1)
    
            ax.plot(time, baseline+hist_model_50, color = 'C1', alpha=1)
            ax.fill_between(time, baseline+hist_model_10, baseline+hist_model_90, color = 'C1', linewidth=0, alpha=0.5)
            ax.fill_between(time, baseline+hist_model_5, baseline+hist_model_95, color = 'C1', linewidth=0, alpha=0.1)
    
            units = 'Forcing $W m^{-2}$'
            if 'Emissions' in targ:
                units = 'Emissions $Tg yr^{-1}$'

            ax.set_ylabel(units)
            ax.set_xlim([1750, 2100])
                
            hist_model_err_50, hist_model_err_90, hist_model_err_95, hist_model_err_10, hist_model_err_5 = np.percentile(hist_model_array - targ_array, [50, 90, 95, 10, 5], axis=1)
    
            hist_err = {}
            hist_err['50'] = hist_model_err_50
            hist_err['5'] = hist_model_err_5
            hist_err['95'] = hist_model_err_95
            hist_err['10'] = hist_model_err_10
            hist_err['90'] = hist_model_err_90

                        
            f = FAIR()
            
            f.define_time(1750, 2100, 1)
            
            scenarios = ['ssp245']
            f.define_scenarios(scenarios)
            
            configs = ['5', '10', '50', '90', '95']
            f.define_configs(configs)
            
            species, properties = read_properties()
            f.define_species(species, properties)
            
            f.allocate()
            f.fill_species_configs()
            f.fill_from_rcmip()
                        
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
    
            for perc in hist_err.keys():
                if 'Emissions' in targ:
                    f.emissions.loc[
                        dict(
                            scenario="ssp245",
                            config=perc,
                            specie=regr_data_for_plot_actual[targ][1]
                        )
                    ] += hist_err[perc]
            
                else:
                    f.forcing.loc[
                        dict(
                            timebounds=f.timebounds,
                            scenario="ssp245",
                            config=perc,
                            specie="Solar"
                        )
                    ] += np.append(hist_err[perc], hist_err[perc][-1])
            f.run()
                
            ax = axs[targ_i, 1]
            ax.axhline(0, linestyle = '--', color = 'grey')
            ax.set_ylabel('K')
            ax.set_xlim([1750, 2100])
            ax.set_ylim([-0.06, 0.06])

    
            ax.plot(time_gmst, 
                    f.temperature.sel(scenario="ssp245", config="50", layer=0) - gmst_ssp245, 
                    color = 'C2', alpha=1, label='Error')
            
            ax.fill_between(time_gmst, 
                    f.temperature.sel(scenario="ssp245", config="10", layer=0) - gmst_ssp245, 
                    f.temperature.sel(scenario="ssp245", config="90", layer=0) - gmst_ssp245, 
                    color = 'C2', linewidth=0, alpha=0.5)
            
            ax.fill_between(time_gmst, 
                    f.temperature.sel(scenario="ssp245", config="5", layer=0) - gmst_ssp245, 
                    f.temperature.sel(scenario="ssp245", config="95", layer=0) - gmst_ssp245, 
                    color = 'C2', linewidth=0, alpha=0.1)
    
handles = [
    Line2D([0], [0], color='C0', lw=2, label='Target'),
    Line2D([0], [0], color='C1', lw=2, label='Historical model'),

    Patch(facecolor='black', alpha=0.5, label='10-90%'),
    Patch(facecolor='black', alpha=0.1, label='5-95%'),
]

axs[0,0].legend(handles=handles, loc='upper left')

handles = [
    Line2D([0], [0], color='C2', lw=2, label='Error'),
    Patch(facecolor='C2', alpha=0.5, label='10-90%'),
    Patch(facecolor='C2', alpha=0.1, label='5-95%'),
]

axs[0,1].legend(handles=handles, loc='lower left')


fig.tight_layout()

plt.savefig(f'{figdir}/S1.png', dpi=300)

