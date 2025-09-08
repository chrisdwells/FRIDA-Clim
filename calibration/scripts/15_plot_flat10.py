import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats

load_dotenv()

GtC_to_MtCO2 = 3670
years = 500
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))


def loaddata(df, n_years, members, varname, offset=False):
    var_data = np.full((n_years, members), np.nan)
    for i in np.arange(members):
        var_data[:,i] = df[f'="Run {i+1}: {varname}"'][:n_years]
    if offset == True:
        var_data = var_data - var_data[0,:]
    return var_data

def plot_reversibility(axs, df, x, color, var, varname, units='', cumul=False):
        
    var_data = loaddata(df, 300, output_ensemble_size, var, offset=True)
    
    if cumul == True:
        var_data = np.cumsum(var_data, axis=0)
    
    axs.plot(x[:150], np.median(var_data[:150,:], axis=1), color =color, linestyle = '--')
    axs.plot(x[150:], np.median(var_data[150:,:], axis=1), color =color, label = varname)
    axs.fill_between(x, np.percentile(var_data, 5, axis=1), 
                       np.percentile(var_data, 95, axis=1), 
                       color = color, alpha=0.3, linewidth=0)
    
    axs.set_xlabel('Cumul. CO2 emissions (GtC)')
    axs.set_ylabel(units)

def plot_dist(axs, arr, x1, x2, varname, units):
    
    var_dist = scipy.stats.gaussian_kde(arr)
    xs = np.linspace(x1, x2, 100)

    axs.plot(xs, var_dist(xs))
    
    axs.set_title(varname)
    axs.set_xlabel(units)

scens = {
    "flat10":['grey', 'solid', [0, 100]],
    "flat10-zec":['C1', 'solid', [100, 300]],
    "flat10-cdr":['C0', 'solid', [100, 300]],
    "flat10-nz":['C2', '--', [100, 300]],
    }


#%%

# fig 1

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
ax = ax.ravel()

for scen in scens.keys():
    ems_in = pd.read_csv(f'../../frida_clim_processing/data/processed_for_frida/emissions_{scen.replace("-", "_")}.csv')
    ems_gtc = ems_in['Emissions.CO2 Emissions from Fossil use'][:years]/GtC_to_MtCO2
    
    ax[0].plot(ems_in['Year'][:years], ems_gtc,
               color=scens[scen][0], linestyle = scens[scen][1], label=f'{scen}')

    if scen == "flat10-cdr":
        ems_cumul_cdr = np.cumsum(ems_gtc)[:300]
    

ax[0].legend()
ax[0].axhline(y=0, color='grey', linestyle='--')    
ax[0].set_ylabel('Emissions (GtC/yr)')
ax[0].set_xlabel('Year')
ax[0].set_xlim([0,320])
ax[0].set_ylim([-15,15])


df_cdr = pd.read_csv('../data/flat10_output/flat10-cdr_output.csv')

plot_reversibility(ax[1], df_cdr, ems_cumul_cdr, 'C0', "CO2 Forcing.Atmospheric CO2 Concentration[1]", 'Atmos conc. (ppm)')
plot_reversibility(ax[1], df_cdr, ems_cumul_cdr, 'C1', "Terrestrial Carbon Balance.Terrestrial carbon balance[1]", 'Land sink', cumul=True)
plot_reversibility(ax[1], df_cdr, ems_cumul_cdr, 'C2', "Ocean.Total ocean carbon", 'Ocean sink')

handles, labels = ax[1].get_legend_handles_labels()

handles.append(Line2D([0], [0], color='grey', linestyle='--'))
labels.append('Ramp up')

handles.append(Line2D([0], [0], color='grey'))
labels.append('Ramp down')

ax[1].legend(handles, labels)
ax[1].set_xlabel('Cumul. CO2 emissions (GtC)')
ax[1].set_ylabel('GtC, ppm')

plot_reversibility(ax[2], df_cdr, ems_cumul_cdr, 'C0', 
                   "Energy Balance Model.Land & Ocean Surface Temperature[1]", 'GMST')

ax[2].set_xlabel('Cumul. CO2 emissions (GtC)')
ax[2].set_ylabel('K')


df_flat10 = pd.read_csv('../data/flat10_output/flat10_output.csv')

gmst_flat10 = loaddata(df_flat10, 500, output_ensemble_size, "Energy Balance Model.Land & Ocean Surface Temperature[1]")

t100 = np.mean(gmst_flat10[91:110,:], axis=0)
plot_dist(ax[3], t100, 0, 4, 'T100yr', 'K')


df_zec = pd.read_csv('../data/flat10_output/flat10-zec_output.csv')

gmst_zec = loaddata(df_zec, 500, output_ensemble_size, "Energy Balance Model.Land & Ocean Surface Temperature[1]")

zec100 = np.mean(gmst_zec[191:210,:], axis=0) - np.mean(gmst_zec[91:110,:], axis=0)
plot_dist(ax[4], zec100, -0.7, 0.7, 'ZEC100', 'K')

zec300 = np.mean(gmst_zec[291:310,:], axis=0) - np.mean(gmst_zec[91:110,:], axis=0)
plot_dist(ax[5], zec300, -0.7, 0.7, 'ZEC300', 'K')


gmst_cdr = loaddata(df_cdr, 500, output_ensemble_size, "Energy Balance Model.Land & Ocean Surface Temperature[1]")

peak_idxs = np.argmax(gmst_cdr, axis=0)
tPW = df_cdr['Year'][peak_idxs] - 150
plot_dist(ax[6], tPW, -25, 25, 't-PW', 'Years')

tr1000 = np.mean(gmst_cdr[191:210,:], axis=0) - np.mean(gmst_flat10[91:110,:], axis=0)
plot_dist(ax[7], tr1000, -1, 1, 'TR1000', 'K')

tr0 = np.mean(gmst_cdr[301:320,:], axis=0)
plot_dist(ax[8], tr0, -1, 1, 'TR0', 'K')

plt.tight_layout()
plt.savefig(
    "../plots/flat10/flat10_fig1.png"
)

#%%

# fig2

def carbon_budget_plot(df, axs, xlim, legend=False):
    
    atmos_scen = np.median(np.gradient(loaddata(df, 500, output_ensemble_size, "CO2 Forcing.Atmospheric CO2 Concentration[1]", 
                          offset=True)*ppm_to_GtC, axis=0), axis=1)
    ocean_scen =  np.median(np.gradient(loaddata(df, 500, output_ensemble_size, "Ocean.Total ocean carbon"), axis=0), axis=1)
    land_scen =  np.median(loaddata(df, 500, output_ensemble_size, "Terrestrial Carbon Balance.Terrestrial carbon balance[1]"), axis=1)
    
    components = [land_scen, ocean_scen, atmos_scen]
    labels = ['Land', 'Ocean', 'Atmosphere']
    colors = ['#2ca02c', '#4b0082', '#87ceeb']
    
    pos_base = np.zeros_like(df['Year'][:500], dtype=float)
    neg_base = np.zeros_like(df['Year'][:500], dtype=float)
    
    for comp, label, color in zip(components, labels, colors):
        comp = np.array(comp)
        
        pos_part = np.where(comp > 0, comp, 0)
        neg_part = np.where(comp < 0, comp, 0)
    
        axs.fill_between(df['Year'][:500], pos_base, pos_base + pos_part, facecolor=color, label=label)
        pos_base += pos_part  
        
        axs.fill_between(df['Year'][:500], neg_base, neg_base + neg_part, facecolor=color, hatch='...')
        neg_base += neg_part  
        
    axs.axhline(0, color='black', linewidth=1)

    total = atmos_scen + ocean_scen + land_scen
    axs.plot(df['Year'][:500], total, color='black', linewidth=2, label='Total')
    
    axs.set_xlim(xlim)
    axs.set_ylim([-11,11])
    axs.set_xlabel('Years')
    axs.set_ylabel('GtC/yr')

    if legend == True:
        axs.legend()

ppm_to_GtC = 2.13

fig, ax = plt.subplots(3, 4, figsize=(20, 15))

for s_i, scen in enumerate(scens.keys()):
    df_scen = pd.read_csv(f'../data/flat10_output/{scen}_output.csv')
    
    gmst_scen = loaddata(df_scen, 500, output_ensemble_size, "Energy Balance Model.Land & Ocean Surface Temperature[1]")
    
    ax[0, s_i].plot(df_scen['Year'][:500], np.median(gmst_scen, axis = 1))
    ax[0, s_i].fill_between(df_scen['Year'][:500], np.percentile(gmst_scen, 5, axis = 1),
                            np.percentile(gmst_scen, 95, axis = 1), linewidth=0, alpha=0.3)

    ax[0, s_i].set_title(f'{scen} GMST')
    ax[0, s_i].set_xlabel('Years')
    ax[0, s_i].set_ylabel('K')

    legend = False
    
    ems_in = pd.read_csv(f'../../frida_clim_processing/data/processed_for_frida/emissions_{scen.replace("-", "_")}.csv')
    ems_gtc = ems_in['Emissions.CO2 Emissions from Fossil use'][:years]/GtC_to_MtCO2
    ems_cumul = np.cumsum(ems_gtc)[:300]
        
    if scen == 'flat10-zec':
        legend=True
    carbon_budget_plot(df_scen, ax[1, s_i], scens[scen][2], legend=legend)
    
    plot_reversibility(ax[2, s_i], df_scen, ems_cumul, 'C0', 
                        "Energy Balance Model.Land & Ocean Surface Temperature[1]", 'GMST', units='K')

plt.tight_layout()
plt.savefig(
    "../plots/flat10/flat10_fig2.png"
)

