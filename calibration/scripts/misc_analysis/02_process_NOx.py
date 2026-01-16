import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import chain, combinations

# This script looks at which species can be used to best predict 
# NOx emissions.

# We split NOx into AFOLU and non-AFOLU components, since they have different
# drivers.

# For future scenarios, the emissions aren't always harmonised, so we 
# normalise the emissions in 2015.


datadir = 'data'
figdir = 'figures'

# Organise historical NOx emissions, and use Total and AFOLU to generate non-AFOLU

nox_hist_ems = {}

# from https://zenodo.org/records/4589756
df_hist = pd.read_csv(f"{datadir}/rcmip-emissions-annual-means-v5-1-0.csv")

df_hist_world = df_hist.loc[(df_hist['Region'] == "World") & (df_hist["Scenario"]=='historical')]

vars_to_plot = {
    "Emissions|NOx":"black",
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning":"blue",
    "Emissions|NOx|MAGICC AFOLU|Agriculture":"red",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning":"green",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning":"yellow",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning":"purple",
    "Emissions|NOx|MAGICC Fossil and Industrial":"orange",
    }

gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]

gfed_factor = 46.006/30.006

for index, row in df_hist_world.iterrows():
    
    if row["Variable"] in vars_to_plot.keys():

        plt.plot(1750+np.arange(751), row["1750":"2500"].values, 
             label=row["Variable"], color=vars_to_plot[row["Variable"]])

        if row["Variable"] in gfed_sectors:
            plt.plot(1750+np.arange(751), row["1750":"2500"].values*gfed_factor, 
                 color=vars_to_plot[row["Variable"]],
                 linestyle = 'dashed')

nox_hist_ems["NOx|AFOLU"]  = (
    df_hist.loc[
        (df_hist["Scenario"] == "historical")
        & (df_hist["Region"] == "World")
        & (df_hist["Variable"].isin(gfed_sectors)),
        "1750":"2014",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    .sum(axis=0)*gfed_factor

    + df_hist.loc[
        (df_hist["Scenario"] == "historical")
        & (df_hist["Region"] == "World")
        & (df_hist["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
        "1750":"2014",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)


nox_hist_ems["NOx"] = (
    nox_hist_ems["NOx|AFOLU"]
    + df_hist.loc[
        (df_hist["Scenario"] == "historical")
        & (df_hist["Region"] == "World")
        & (df_hist["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
        "1750":"2014",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)

plt.plot(1750+np.arange(265), nox_hist_ems["NOx"], 
     color='black', linestyle = 'dashed', label='Full NOx')

plt.legend(fontsize=8)

nox_hist_ems["NOx|non-AFOLU"] = nox_hist_ems["NOx"] - nox_hist_ems["NOx|AFOLU"]


vars_to_plot = [
    "Emissions|N2O",
    "Emissions|N2O|MAGICC AFOLU",
    "Emissions|N2O|MAGICC Fossil and Industrial"
    ]

for index, row in df_hist_world.iterrows():
    
    # if row["Variable"] in vars_to_plot:
    if "N2O" in row["Variable"]:
        plt.plot(1750+np.arange(751), row["1750":"2500"].values, label=row["Variable"])
        print(row["Variable"])
        print(row["1980"])


plt.legend()


# Organise future emissions

df = pd.read_csv(f"{datadir}/AR6_Scenarios_Database_World_ALL_CLIMATE_v1.1.csv")

# get full set of scenarios

scenarios_list = []    
for index, row in df.iterrows():
    scenarios_list.append(row[1])
    
scenarios_set = set(scenarios_list)
scenarios_set_list = list(scenarios_set)

# get dictionary with the models for each scenario
# needs to have the infilled total emissions to normalise future emissions..

scens_models = {}
for scen in scenarios_set_list:
    if scen not in scens_models.keys():
        scen_data = df.loc[(df['Scenario'] == scen),"Model"]
        scen_model_list = set(scen_data.values)
        scens_models[scen] = scen_model_list


species_ems = {
    'NOx':"Emissions|NOx", 
    'N2O':"Emissions|N2O", 
    'NOx|AFOLU':"Emissions|NOx|MAGICC AFOLU", 
    'N2O|AFOLU':"Emissions|N2O|MAGICC AFOLU",
    'CH4':"Emissions|CH4",
    'Sulfur':"Emissions|Sulfur",
    }

species_ems_full = ['Emissions|NOx', 'Emissions|N2O', 'Emissions|NOx|AFOLU', 'Emissions|N2O|AFOLU',
                    "AR6 climate diagnostics|Infilled|Emissions|NOx", 
                    "AR6 climate diagnostics|Infilled|Emissions|NOx"
                    ]

scen_models_with_all_ems = {}
for scen in scens_models.keys():
    print('\n')
    print(scen)
    models_with_all_ems = []
    for model in scens_models[scen]:
        print(model)
        scen_array = df.loc[(df['Scenario'] == scen) & (df['Model'] == model)]
        
        common_specs = []
        for index, row in scen_array.iterrows():
            if "Emissions|" in row[3]:
                file_spec = row[3]
                if file_spec in species_ems_full:
                    common_specs.append(file_spec)
        ok_flag = 1
        for spec in species_ems_full:
            if spec not in common_specs:
                print(f'Missing: {spec}')
                ok_flag = 0 
        if ok_flag == 1:
            models_with_all_ems.append(model)
    scen_models_with_all_ems[scen] = models_with_all_ems
    
total_nr_models_scens = 0
for scen in scens_models.keys():
    total_nr_models_scens += len(scens_models[scen])

total_nr_models_scens_with_all_ems = 0
for scen in scens_models.keys():
    total_nr_models_scens_with_all_ems += len(scen_models_with_all_ems[scen])
        
print(f'{np.around(100*total_nr_models_scens_with_all_ems/total_nr_models_scens, decimals=0)} % of scenario-model pairs have all emissions')


 #%%
 
# generate array of historical 

scen_model_full_list = []
for scen in scen_models_with_all_ems.keys():
    for model in scen_models_with_all_ems[scen]:
        scen_model = scen + '___' + model
        scen_model_full_list.append(scen_model)


emis_dict = {}
for specie in species_ems.keys():
    if "NOx" not in specie:
        emis_dict[specie] = np.zeros((351, len(scen_model_full_list)))
        for sm_i, scen_model in enumerate(scen_model_full_list):
            emis_dict[specie][:265,sm_i] = df_hist_world.loc[df_hist_world[
                'Variable'] == species_ems[specie]].values[0,7:272]

emis_dict["NOx|AFOLU"] = np.zeros((351, len(scen_model_full_list)))
for sm_i, scen_model in enumerate(scen_model_full_list):
    emis_dict["NOx|AFOLU"][:265,sm_i] = nox_hist_ems["NOx|AFOLU"]


emis_dict["NOx"] = np.zeros((351, len(scen_model_full_list)))
for sm_i, scen_model in enumerate(scen_model_full_list):
    emis_dict["NOx"][:265,sm_i] = nox_hist_ems["NOx"]


#%%

# normalise future emissions

for sm_i, scen_model in enumerate(scen_model_full_list):
    
    print(f'{sm_i}: {scen_model}')
    scen, model = scen_model.split("___")
        
    
    for specie in species_ems.keys():
        emis_in = df.loc[(df['Scenario'] == scen) & 
                         (df['Model'] == model) & 
                         (df['Variable'] == "Emissions|" + specie),
                         "2015":"2100"].interpolate(axis=1).values.squeeze()
        
        factor_2014 = emis_dict[specie][264,0]
        
        emis_in_norm = emis_in*(factor_2014/emis_in[0])

        emis_dict[specie][265:,sm_i] = emis_in_norm
        
            
emis_dict['Scenarios'] = scen_model_full_list

timepoints = 1750+np.arange(351)



#%%

# remove if future emissions do something radically different - as this 
# strongly alters the fit
# just apply this to N2O, NOx
species_ems_reduced = ['NOx', 'N2O', 'NOx|AFOLU', 'N2O|AFOLU']

sm_i_list = []
for specie in species_ems_reduced:
    print(specie)
    for sm_i, scen_model in enumerate(scen_model_full_list):
        if np.amax(emis_dict[specie][265:,sm_i]) > 5*emis_dict[specie][265,sm_i
           ] or np.amin(emis_dict[specie][265:,sm_i]) < 0.1*emis_dict[specie][265,sm_i
              ] or np.isnan(emis_dict[specie][265,sm_i]) == True:
            if sm_i not in sm_i_list:
                sm_i_list.append(sm_i)

keep_idxs = []
for sm_i, scen_model in enumerate(scen_model_full_list):
    if sm_i not in sm_i_list:
        keep_idxs.append(sm_i)

keep_idxs = np.asarray(keep_idxs)

scen_model_full_list_cropped = copy.deepcopy(scen_model_full_list)

emis_dict_cropped = copy.deepcopy(emis_dict)

scen_model_full_list_cropped = [j for i, j in enumerate(scen_model_full_list) if i not in sm_i_list]

for specie in species_ems:
    emis_dict_cropped[specie] = emis_dict[specie][:,keep_idxs]
            
emis_dict_cropped['Scenarios'] = scen_model_full_list_cropped
    
        #%%
        
fig, axs = plt.subplots(2, 2)

for s_i, specie in enumerate(species_ems_reduced):

    ax = plt.subplot(2, 2, s_i+1)
    
    emis_50, emis_90, emis_10 = np.percentile(emis_dict_cropped[specie], [50, 90, 10], axis=1)

    ax.plot(timepoints, emis_50, color='blue')
    ax.fill_between(timepoints, emis_10, emis_90, color='blue', linewidth=0, alpha=0.5)
    ax.axhline(emis_50[0], color='grey', linestyle = '--')

    ax.set_title(specie)
    
plt.tight_layout()
    # axs[s_i].set_xlim([2010, 2030])
# plt.ylim([0, 70000])

#%%

# now define the potential drivers of these NOx emissions, and output the emissions

emis_dict_cropped['NOx|non-AFOLU'] = emis_dict_cropped['NOx'] - emis_dict_cropped['NOx|AFOLU']
emis_dict_cropped['N2O|non-AFOLU'] = emis_dict_cropped['N2O'] - emis_dict_cropped['N2O|AFOLU']

potential_links = {
    'NOx|AFOLU':['N2O|non-AFOLU', 'N2O|', 'N2O|AFOLU', 'CH4|', 'Sulfur|'],
    'NOx|non-AFOLU':['N2O|non-AFOLU', 'N2O|', 'N2O|AFOLU'],
    }

vars_calc_future = {
    'NOx|AFOLU':emis_dict_cropped['NOx|AFOLU'][265:,:] - emis_dict_cropped['NOx|AFOLU'][0,:],
    'NOx|non-AFOLU':emis_dict_cropped['NOx|non-AFOLU'][265:,:] - emis_dict_cropped['NOx|non-AFOLU'][0,:],
    'NOx|':emis_dict_cropped['NOx'][265:,:] - emis_dict_cropped['NOx'][0,:],

    'N2O|AFOLU':emis_dict_cropped['N2O|AFOLU'][265:,:] - emis_dict_cropped['N2O|AFOLU'][0,:],
    'N2O|non-AFOLU':emis_dict_cropped['N2O|non-AFOLU'][265:,:] - emis_dict_cropped['N2O|non-AFOLU'][0,:],
    'N2O|':emis_dict_cropped['N2O'][265:,:] - emis_dict_cropped['N2O'][0,:],

    'CH4|':emis_dict_cropped['CH4'][265:,:] - emis_dict_cropped['CH4'][0,:],
    'Sulfur|':emis_dict_cropped['Sulfur'][265:,:] - emis_dict_cropped['Sulfur'][0,:],
    }
    
vars_calc_hist = {
    'NOx|AFOLU':emis_dict_cropped['NOx|AFOLU'][:265,0] - emis_dict_cropped['NOx|AFOLU'][0,0],
    'NOx|non-AFOLU':emis_dict_cropped['NOx|non-AFOLU'][:265,0] - emis_dict_cropped['NOx|non-AFOLU'][0,0],
    'NOx|':emis_dict_cropped['NOx'][:265,0] - emis_dict_cropped['NOx'][0,0],
    
    'N2O|AFOLU':emis_dict_cropped['N2O|AFOLU'][:265,0] - emis_dict_cropped['N2O|AFOLU'][0,0],
    'N2O|non-AFOLU':emis_dict_cropped['N2O|non-AFOLU'][:265,0] - emis_dict_cropped['N2O|non-AFOLU'][0,0],
    'N2O|':emis_dict_cropped['N2O'][:265,0] - emis_dict_cropped['N2O'][0,0],

    'CH4|':emis_dict_cropped['CH4'][:265,0] - emis_dict_cropped['CH4'][0,0],
    'Sulfur|':emis_dict_cropped['Sulfur'][:265,0] - emis_dict_cropped['Sulfur'][0,0],
    }

full_save_dict = {}
full_save_dict['NOx|non-AFOLU'] = emis_dict_cropped['NOx|non-AFOLU']
full_save_dict['N2O|non-AFOLU'] = emis_dict_cropped['N2O|non-AFOLU']

full_save_dict['NOx|AFOLU'] = emis_dict_cropped['NOx|AFOLU']
full_save_dict['Sulfur'] = emis_dict_cropped['Sulfur']

full_save_dict['Scenarios'] = emis_dict_cropped['Scenarios']

with open(f'{datadir}/scenario_info_n2o_nox.pkl', 'wb') as handle:
    pickle.dump(full_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#%%

# # apply the fits to each plausible predictor, for the historical period and 
# # across all future scenarios

# regr = LinearRegression(fit_intercept=False)
# def powerset(iterable):
#     s = list(iterable)
#     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# n_scens = len(scen_model_full_list_cropped)

# regr_data_for_plot = {}

# for targ in potential_links.keys():
#     regr_data_for_plot[targ] = {}
    
#     preds = powerset(potential_links[targ])
#     preds_list = []
#     for pred in preds:
#         pred_list = []
#         for p_i in np.arange(len(pred)):
#             pred_list.append(pred[p_i])
#         if len(pred_list) > 0:
#             preds_list.append(pred_list)
        
        
#     regr_data_for_plot[targ]['Preds'] = preds_list

#     coef_list = []
#     r2_list = []
#     int_list = []
    
#     for ni in np.arange(len(preds_list)):
#         pred = preds_list[ni]
#         pred_string= ''
#         for p in pred:
#             pred_string = pred_string + ' + ' + p
        
#         regr_data_for_plot[targ][pred_string] = {}
#         pred_data = np.zeros((86, n_scens, len(pred)))
#         for pred_i, pred_component in enumerate(pred):
#             pred_data[:,:,pred_i] = vars_calc_future[pred_component]
#             pred_array = pred_data.reshape(-1, pred_data.shape[-1])
            
        
#         targ_data = vars_calc_future[targ]
#         targ_array = targ_data.flatten()

#         regr.fit(pred_array, targ_array)
        
#         coef_list.append(regr.coef_)
        
#         pred_for_error = regr.predict(pred_array)
        
#         r2_list.append(r2_score(targ_array, pred_for_error))
        
#         int_list.append(regr.intercept_)
        
#         regr_data_for_plot[targ][pred_string]['Future_targ_data'] = targ_data
#         regr_data_for_plot[targ][pred_string]['Future_pred_data'] = pred_data

            
#     regr_data_for_plot[targ]['Future_coefs'] = coef_list
#     regr_data_for_plot[targ]['Future_r2'] = r2_list
#     regr_data_for_plot[targ]['Future_int'] = int_list

    
#     coef_list = []
#     r2_list = []
#     int_list = []
    
#     for ni in np.arange(len(preds_list)):
#         pred = preds_list[ni]
#         pred_string= ''
#         for p in pred:
#             pred_string = pred_string + ' + ' + p
            
#         pred_array = np.zeros((265, len(pred)))
#         for pred_i, pred_component in enumerate(pred):
#             pred_array[:,pred_i] = np.asarray(vars_calc_hist[pred_component]).flatten()
            
        
#         targ_data = vars_calc_hist[targ]
#         targ_array = targ_data.flatten()
        
#         regr.fit(pred_array, targ_array)
        
#         coef_list.append(regr.coef_)
        
#         pred_for_error = regr.predict(pred_array)
        
#         r2_list.append(r2_score(targ_array, pred_for_error))
            
#         int_list.append(regr.intercept_)

#         regr_data_for_plot[targ][pred_string]['Historical_targ_data'] = targ_data
#         regr_data_for_plot[targ][pred_string]['Historical_pred_data'] = pred_array

#     regr_data_for_plot[targ]['Hist_coefs'] = coef_list
#     regr_data_for_plot[targ]['Hist_r2'] = r2_list
#     regr_data_for_plot[targ]['Hist_int'] = int_list



# #%%

# # plot the results

# idx_dict = {}
# for targ in potential_links.keys():
#     preds_list = regr_data_for_plot[targ]['Preds']
#     fig, axs = plt.subplots(4, 1, figsize=(1.8*len(preds_list), 12))
    
#     perc_dif = np.zeros(len(preds_list))
#     for p_i, pred in enumerate(preds_list):
#         perc_dif[p_i] = np.mean(100*(np.abs(regr_data_for_plot[targ]['Future_coefs'][p_i] - regr_data_for_plot[targ]['Hist_coefs'][p_i])/np.abs(np.abs(regr_data_for_plot[targ]['Hist_coefs'][p_i]))))

#     sorted_idxs = np.argsort(perc_dif)
#     idx_dict[targ] = sorted_idxs
    
#     for i_i, idx in enumerate(sorted_idxs):
#         coefs_hist = regr_data_for_plot[targ]['Hist_coefs'][idx]
#         coefs_future = regr_data_for_plot[targ]['Future_coefs'][idx]

#         r2_hist = regr_data_for_plot[targ]['Hist_r2'][idx]
#         r2_future = regr_data_for_plot[targ]['Future_r2'][idx]
        
#         n_pi = len(coefs_hist)
#         delta = (n_pi-1)/2
#         for pi in np.arange(n_pi):
#             if i_i == 0 and pi == 0:
#                 axs[0].plot(i_i + 0.2*(pi-delta), coefs_hist[pi], color = 'blue', marker = '*', label = 'Historical')
#                 axs[0].plot(i_i + 0.2*(pi-delta), coefs_future[pi], color = 'orange', marker = 'X', label = 'Future')

#             else:
#                 axs[0].plot(i_i + 0.2*(pi-delta), coefs_hist[pi], color = 'blue', marker = '*')
#                 axs[0].plot(i_i + 0.2*(pi-delta), coefs_future[pi], color = 'orange', marker = 'X')


#         axs[1].bar(i_i, perc_dif[idx], color = 'red')

#         axs[2].plot(i_i + 0.1*(pi-delta), r2_hist, color = 'blue', marker = '*')
#         axs[2].plot(i_i + 0.1*(pi-delta), r2_future, color = 'orange', marker = 'X')

#         axs[3].bar(i_i + 0.1*(pi-delta), r2_hist - r2_future, color = 'red')

#     axs[0].set_title(f'Target: {targ}. Regression coefficients')
#     axs[1].set_title('Mean regression coefficient percentage change Hist to Future')
#     axs[2].set_title('R2')
#     axs[3].set_title('R2 Hist minus Future')

#     axs[0].set_xlim([-0.5, len(preds_list)-0.5])
#     axs[1].set_xlim([-0.5, len(preds_list)-0.5])
#     axs[2].set_xlim([-0.5, len(preds_list)-0.5])
#     axs[3].set_xlim([-0.5, len(preds_list)-0.5])

#     axs[0].set_xticklabels([''])
#     axs[1].set_xticklabels([''])
#     axs[2].set_xticklabels([''])
    
#     axs[0].axhline(0, linestyle = '--', color = 'grey')


#     sorted_preds = []
#     for idx in sorted_idxs:
#         pred_name_label = ''
#         for pred_single in preds_list[idx]:
#             pred_name_label = pred_name_label + '\n' + pred_single
#         sorted_preds.append(pred_name_label)

#     axs[3].set_xticks(np.arange(len(sorted_preds)))
#     axs[3].set_xticklabels(sorted_preds)

#     axs[0].legend()

#     axs[1].set_ylim([0, 200])


#     fig.tight_layout()

#     plt.savefig(f'{figdir}/regressions/{targ.replace("|", "_")}.png', dpi=300)
#     plt.clf()
    

# #%%

# error_arrays = {}
# for targ in potential_links.keys():
#     error_arrays[targ] = {}
#     preds_list = regr_data_for_plot[targ]['Preds']

#     fig, axs = plt.subplots(3, len(preds_list), figsize=(5*len(preds_list), 10))

#     idxs = idx_dict[targ]

#     for pred_i in np.arange(len(preds_list)):
        
#         axs = plt.subplot(3, len(preds_list), pred_i+1)
#         axs.axhline(0, linestyle = '--', color = 'grey')

#         p_i = idxs[pred_i]

#         pred = preds_list[p_i]
#         pred_string= ''
#         for p in pred:
#             pred_string = pred_string + ' + ' + p
        
#         error_arrays[targ][pred_string] = {}
        
#         pred_name_lines = ''
#         for pred_single in pred:
#             pred_name_lines = pred_name_lines + '\n' + pred_single
#         axs.set_title(f'Targ: {targ}, pred: {pred_name_lines}')
        
#         coefs_hist = regr_data_for_plot[targ]['Hist_coefs'][p_i]
#         intercept_hist = regr_data_for_plot[targ]['Hist_int'][p_i]
        
#         coefs_future = regr_data_for_plot[targ]['Future_coefs'][p_i]
#         intercept_future = regr_data_for_plot[targ]['Future_int'][p_i]
        
#         targ_array = np.zeros((timepoints.shape[0], n_scens))
#         future_model_array = np.zeros((timepoints.shape[0], n_scens))
#         hist_model_array = np.zeros((timepoints.shape[0], n_scens))

#         for scen_i in np.arange(n_scens):
#             targ_timeseries = np.concatenate((regr_data_for_plot[targ][pred_string]['Historical_targ_data'], regr_data_for_plot[targ][pred_string]['Future_targ_data'][:,scen_i]))
#             targ_array[:,scen_i] = targ_timeseries
#             pred_timeseries = np.zeros((timepoints.shape[0], len(pred)))
            
#             hist_model = np.zeros(timepoints.shape[0])
#             hist_model.fill(intercept_hist)
#             for pred_component_i, pred_component in enumerate(pred):
#                 pred_timeseries[:,pred_component_i] = np.concatenate((regr_data_for_plot[targ][pred_string]['Historical_pred_data'][:,pred_component_i], regr_data_for_plot[targ][pred_string]['Future_pred_data'][:,scen_i,pred_component_i]))
#                 hist_model += coefs_hist[pred_component_i]*pred_timeseries[:,pred_component_i]
            

#             hist_model_array[:,scen_i] = hist_model

#             future_model = np.zeros(timepoints.shape[0])
#             future_model.fill(intercept_future)
#             for pred_component_i, pred_component in enumerate(pred):
#                 pred_timeseries[:,pred_component_i] = np.concatenate((regr_data_for_plot[targ][pred_string]['Historical_pred_data'][:,pred_component_i], regr_data_for_plot[targ][pred_string]['Future_pred_data'][:,scen_i,pred_component_i]))
#                 future_model += coefs_future[pred_component_i]*pred_timeseries[:,pred_component_i]
                
#             future_model_array[:,scen_i] = future_model
        
#         targ_50, targ_90, targ_95, targ_10, targ_5 = np.percentile(targ_array, [50, 90, 95, 10, 5], axis=1)
#         future_model_50, future_model_90, future_model_95, future_model_10, future_model_5 = np.percentile(future_model_array, [50, 90, 95, 10, 5], axis=1)
#         hist_model_50, hist_model_90, hist_model_95, hist_model_10, hist_model_5 = np.percentile(hist_model_array, [50, 90, 95, 10, 5], axis=1)


#         if pred_i == 0:
#             axs.set_ylabel(f'{targ}') 
            
#         axs.plot(timepoints, targ_50, color = 'black', alpha=1, label='Target')
#         axs.fill_between(timepoints, targ_10, targ_90, color = 'black', linewidth=0, alpha=0.5)
#         axs.fill_between(timepoints, targ_5, targ_95, color = 'black', linewidth=0, alpha=0.1)

#         axs.plot(timepoints, future_model_50, color = 'orange', alpha=1, label='Future model')
#         axs.fill_between(timepoints, future_model_10, future_model_90, color = 'orange', linewidth=0, alpha=0.5)
#         axs.fill_between(timepoints, future_model_5, future_model_95, color = 'orange', linewidth=0, alpha=0.1)

#         axs.plot(timepoints, hist_model_50, color = 'blue', alpha=1, label='Hist model')
#         axs.fill_between(timepoints, hist_model_10, hist_model_90, color = 'blue', linewidth=0, alpha=0.5)
#         axs.fill_between(timepoints, hist_model_5, hist_model_95, color = 'blue', linewidth=0, alpha=0.1)

#         axs.legend()
 
#         axs = plt.subplot(3, len(preds_list), pred_i+1+len(preds_list))
#         axs.axhline(0, linestyle = '--', color = 'grey')


#         targ_50, targ_90, targ_95, targ_10, targ_5 = np.percentile(targ_array, [50, 90, 95, 10, 5], axis=1)
#         future_model_err_50, future_model_err_90, future_model_err_95, future_model_err_10, future_model_err_5 = np.percentile(future_model_array - targ_array, [50, 90, 95, 10, 5], axis=1)
#         hist_model_err_50, hist_model_err_90, hist_model_err_95, hist_model_err_10, hist_model_err_5 = np.percentile(hist_model_array - targ_array, [50, 90, 95, 10, 5], axis=1)


#         error_arrays[targ][pred_string]['Future_model'] = {}
#         error_arrays[targ][pred_string]['Future_model']['50'] = future_model_err_50
#         error_arrays[targ][pred_string]['Future_model']['90'] = future_model_err_90
#         error_arrays[targ][pred_string]['Future_model']['10'] = future_model_err_10
#         error_arrays[targ][pred_string]['Future_model']['5'] = future_model_err_5
#         error_arrays[targ][pred_string]['Future_model']['95'] = future_model_err_95
        

#         axs.plot(timepoints, hist_model_err_50, color = 'blue', alpha=1, label='Hist model median')
#         axs.fill_between(timepoints, hist_model_err_10, hist_model_err_90, color = 'blue', linewidth=0, alpha=0.5, label='10-90%ile')
#         axs.fill_between(timepoints, hist_model_err_5, hist_model_err_95, color = 'blue', linewidth=0, alpha=0.1, label='5-95%ile')

#         if pred_i == 0:
#             axs.set_ylabel(f'{targ} error \n trained on Hist')
#         axs.legend()

#         axs = plt.subplot(3, len(preds_list), pred_i+1+2*len(preds_list))
#         axs.axhline(0, linestyle = '--', color = 'grey')


#         axs.plot(timepoints, future_model_err_50, color = 'orange', alpha=1, label='Future model median')
#         axs.fill_between(timepoints, future_model_err_10, future_model_err_90, color = 'orange', linewidth=0, alpha=0.5, label='10-90%ile')
#         axs.fill_between(timepoints, future_model_err_5, future_model_err_95, color = 'orange', linewidth=0, alpha=0.1, label='5-95%ile')
  
#         if pred_i == 0:
#             axs.set_ylabel(f'{targ} error \n trained on Future') 
#         axs.legend()
        

#     fig.tight_layout()
    
#     plt.savefig(f'{figdir}/regressions/emulation_error_{targ.replace("|", "_")}.png', dpi=300)
#     plt.clf()

# #%%

# with open(f'{datadir}/outputs/regression_data_NOx.pkl', 'wb') as handle:
#     pickle.dump(regr_data_for_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    