import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
import pickle
from matplotlib.lines import Line2D

# for RCMIP3: bring in their 6 variables as before. 
# Also, keep the aerosol components and ECS.
# Subtract aerosol pi baseline (fix in main?).

# Adapted from FaIR calibrate
# for FRIDA, calculate ECS from the parameters.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
output_ensemble_size=70
calibration = os.getenv("CALIBRATION")

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

valid_temp_flux = np.loadtxt(
    "../../data/constraining/runids_rmse_pass.csv",
).astype(np.int64)

input_ensemble_size = len(valid_temp_flux)

assert input_ensemble_size > output_ensemble_size

# need to manipulate temperature so do differently to others
df_temp = pd.read_csv("../../data/priors_output/priors_temperature.csv")

# GMST now 2014-2023 cf 1850-1900
temp_pi = np.average(df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=1900)].drop(columns='Year').values, axis=0)
temp_pd = np.average(df_temp.loc[(df_temp['Year']>=2014) & (df_temp['Year']<=2023)].drop(columns='Year').values, axis=0)
temp_in = temp_pd - temp_pi

# for others, pull in data by run
# OHC still 2020 minus 1971
df_ohc = pd.read_csv("../../data/priors_output/priors_ocean_heat_content.csv")
ohc_data = np.full((2, samples), np.nan)
for i in np.arange(samples):
    ohc_data[:,i] = df_ohc[f'="Run {i+1}: Energy Balance Model.ocean heat content change[1]"']
# 1000 for units; 0.91 to convert N (what we actual have here) to OHC as per protocol
ohc_in = 0.91*(ohc_data[1,:] - ohc_data[0,:])*1000 

# CO2 2014-2023 average now
df_co2 = pd.read_csv("../../data/priors_output/priors_CO2.csv")
co2_in = np.full(samples, np.nan)
for i in np.arange(samples):
    co2_in[i] = np.mean(df_co2[f'="Run {i+1}: CO2 Forcing.Atmospheric CO2 Concentration[1]"'])
    
# ocean carbon flux 2014-2023 average
df_ocean_co2 = pd.read_csv("../../data/priors_output/priors_ocean_CO2_flux.csv")
df_ocean_co2 = df_ocean_co2.loc[(df_ocean_co2['Year']>=2014) & (df_ocean_co2['Year']<=2023)]
ocean_co2_in = np.full(samples, np.nan)
for i in np.arange(samples):
    ocean_co2_in[i] = np.mean(df_ocean_co2[f'="Run {i+1}: Ocean.Air sea co2 flux[1]"'])
    
# land carbon flux 2014-2023 average
df_land_co2 = pd.read_csv("../../data/priors_output/priors_land.csv")
df_land_co2 = df_land_co2.loc[(df_land_co2['Year']>=2014) & (df_land_co2['Year']<=2023)]
land_co2_in = np.full(samples, np.nan)
for i in np.arange(samples):
    land_co2_in[i] = np.mean(df_land_co2[f'="Run {i+1}: Emissions.land carbon sink[1]"'])
    
# aerosol still 2005-2014
df_aer = pd.read_csv("../../data/priors_output/priors_aerosols.csv")

faci_in = np.full(samples, np.nan)
fari_in = np.full(samples, np.nan)
for i in np.arange(samples):
    faci_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol Cloud Interactions[1]"'])
    fari_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol Radiation Interactions[1]"'])
        
faer_in = fari_in + faci_in

# and keep ECS, TCR
df_ecs_tcr = pd.read_csv(f"../../data/external/samples_for_priors/ecs_tcs_{samples}.csv")
ecs_in = df_ecs_tcr['ecs']
tcr_in = df_ecs_tcr['tcr']

# ensure shape is as we expect
assert temp_in.shape == (samples,)
assert ohc_in.shape == (samples,)
assert co2_in.shape == (samples,)
assert ocean_co2_in.shape == (samples,)
assert land_co2_in.shape == (samples,)
assert faer_in.shape == (samples,)

assert ecs_in.shape == (samples,)
assert tcr_in.shape == (samples,)

#%%
def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    # print(q05, q50, q95, x)
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)

# use this skewnorm approach for all 6 RCMIP variables, 
# given the distributions from RCMIP aren't symmetrical

constraints = [
    'Global Mean Surface Temperature (GMST)',
    'Ocean Heat Content|Global|Total',
    'Atmospheric Concentrations|CO2',
    'Carbon Flux to Oceans',
    'Carbon Flux to Land',
    'Effective Radiative Forcing|Aerosols',
    # 'ERFaci',
    # 'ERFari',
    'ECS',
    'TCR',
        ]

samples_dict = {}

# pull in RCMIP values for these 6
df_rcmip_constraints = pd.read_csv(
    f'{indir}/rcmip_phase3_constraint_targets_with_uncertainty_{rcmip_version}.csv')

for c_i, constraint in enumerate(constraints):
    if constraint not in ['ERFaci', 'ERFari', 'ECS', 'TCR']:
        df_con = df_rcmip_constraints.loc[df_rcmip_constraints["Variable"] == constraint]
        
        params = scipy.optimize.root(opt, [1, 1, 1], 
                 args=(df_con["Lower_bound"].values[0], df_con["Central_estimate"].values[0], df_con["Upper_bound"].values[0])).x
        
        samples_dict[constraint] = scipy.stats.skewnorm.rvs(
            params[0],
            loc=params[1],
            scale=params[2],
            size=10**5,
            random_state=91603 + 1000*c_i,
        )
        
# others as before
ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x

samples_dict["ECS"] = scipy.stats.skewnorm.rvs(
    ecs_params[0],
    loc=ecs_params[1],
    scale=ecs_params[2],
    size=10**5,
    random_state=91603,
)
samples_dict["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
# samples_dict["ERFari"] = scipy.stats.norm.rvs(
#     loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
# )
# samples_dict["ERFaci"] = scipy.stats.norm.rvs(
#     loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
# )


ar_distributions = {}
for constraint in constraints:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples_dict[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples_dict[constraint]

accepted = pd.DataFrame(
    {
     
    'Global Mean Surface Temperature (GMST)': temp_in[valid_temp_flux],
    'Ocean Heat Content|Global|Total': ohc_in[valid_temp_flux],
    'Atmospheric Concentrations|CO2': co2_in[valid_temp_flux],
    'Carbon Flux to Oceans': ocean_co2_in[valid_temp_flux],
    'Carbon Flux to Land': land_co2_in[valid_temp_flux],
    'Effective Radiative Forcing|Aerosols': faer_in[valid_temp_flux],
    # 'ERFaci': faci_in[valid_temp_flux],
    # 'ERFari': fari_in[valid_temp_flux],
    'ECS': ecs_in[valid_temp_flux],
    'TCR': tcr_in[valid_temp_flux],
    },
    index=valid_temp_flux,
)


def calculate_sample_weights(distributions, samples, niterations=50):
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in range(niterations):
        gofs.append([])
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=30
)

effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

assert effective_samples >= output_ensemble_size

draws = []
drawn_samples = accepted.sample(
    n=output_ensemble_size, replace=False, weights=weights, random_state=10099
)
draws.append((drawn_samples))

#%%

target_temp = scipy.stats.gaussian_kde(samples_dict['Global Mean Surface Temperature (GMST)'])
prior_temp = scipy.stats.gaussian_kde(temp_in)
post1_temp = scipy.stats.gaussian_kde(temp_in[valid_temp_flux])
post2_temp = scipy.stats.gaussian_kde(draws[0]['Global Mean Surface Temperature (GMST)'])

target_ohc = scipy.stats.gaussian_kde(samples_dict['Ocean Heat Content|Global|Total'])
prior_ohc = scipy.stats.gaussian_kde(ohc_in)
post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp_flux])
post2_ohc = scipy.stats.gaussian_kde(draws[0]['Ocean Heat Content|Global|Total'])

target_co2 = scipy.stats.gaussian_kde(samples_dict['Atmospheric Concentrations|CO2'])
prior_co2 = scipy.stats.gaussian_kde(co2_in)
post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp_flux])
post2_co2 = scipy.stats.gaussian_kde(draws[0]['Atmospheric Concentrations|CO2'])
post2_co2 = scipy.stats.gaussian_kde(co2_in[draws[0].index])

target_oce_flux = scipy.stats.gaussian_kde(samples_dict['Carbon Flux to Oceans'])
prior_oce_flux = scipy.stats.gaussian_kde(ocean_co2_in)
post1_oce_flux = scipy.stats.gaussian_kde(ocean_co2_in[valid_temp_flux])
post2_oce_flux = scipy.stats.gaussian_kde(draws[0]['Carbon Flux to Oceans'])

target_land_flux = scipy.stats.gaussian_kde(samples_dict['Carbon Flux to Land'])
prior_land_flux = scipy.stats.gaussian_kde(land_co2_in)
post1_land_flux = scipy.stats.gaussian_kde(land_co2_in[valid_temp_flux])
post2_land_flux = scipy.stats.gaussian_kde(draws[0]['Carbon Flux to Land'])

target_aer = scipy.stats.gaussian_kde(samples_dict['Effective Radiative Forcing|Aerosols'])
prior_aer = scipy.stats.gaussian_kde(faer_in)
post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp_flux])
post2_aer = scipy.stats.gaussian_kde(draws[0]['Effective Radiative Forcing|Aerosols'])


target_ecs = scipy.stats.gaussian_kde(samples_dict["ECS"])
prior_ecs = scipy.stats.gaussian_kde(ecs_in)
post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp_flux])
post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples_dict["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp_flux])
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])


dict_distributions = {}

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

fig, ax = plt.subplots(3, 3, figsize=(9, 9))

def dist_plot(axs, start, stop, target, priors, post1, post2, ylims, title, units, dist_name):
    axs.plot(
        np.linspace(start, stop, 1000),
        target(np.linspace(start, stop, 1000)),
        color=colors["target"],
    )
    axs.plot(
        np.linspace(start, stop, 1000),
        priors(np.linspace(start, stop, 1000)),
        color=colors["prior"],
    )
    axs.plot(
        np.linspace(start, stop, 1000),
        post1(np.linspace(start, stop, 1000)),
        color=colors["post1"],
    )
    axs.plot(
        np.linspace(start, stop, 1000),
        post2(np.linspace(start, stop, 1000)),
        color=colors["post2"],
    )
    axs.set_xlim(start, stop)
    axs.set_ylim(ylims[0], ylims[1])
    axs.set_title(title)
    axs.set_yticklabels([])
    axs.set_xlabel(units)
        
    dict_distributions[dist_name] = {}
    dict_distributions[dist_name]['Target'] = target(np.linspace(start, stop, 1000))
    dict_distributions[dist_name]['Priors'] = priors(np.linspace(start, stop, 1000))
    dict_distributions[dist_name]['Post1'] = post1(np.linspace(start, stop, 1000))
    dict_distributions[dist_name]['Post2'] = post2(np.linspace(start, stop, 1000))
    dict_distributions[dist_name]['Xs'] = np.linspace(start, stop, 1000)
    dict_distributions[dist_name]['xlim'] = [start, stop]


dist_plot(ax[0,0], 0.8, 2.0, target_temp, prior_temp, post1_temp, post2_temp,
          [0, 5], "Temperature anomaly", "°C, 2014-2022 minus 1850-1900", 'Temp')
    
dist_plot(ax[0,1], 0, 800, target_ohc, prior_ohc, post1_ohc, post2_ohc,
          [0, 0.006], "Ocean heat content change", "ZJ, 2020 minus 1971", 'OHC')
    
dist_plot(ax[0,2], 400, 420, target_co2, prior_co2, post1_co2, post2_co2,
          [0, 0.3], "CO$_2$ concentration", "ppm, 2014-2023", 'CO2')
    
dist_plot(ax[1,0], -3, 0, target_aer, prior_aer, post1_aer, post2_aer,
          [0, 1], "Aerosol ERF", "W m$^{-2}$, 2005-2014 minus 1850-1900", 'Aerosol')

dist_plot(ax[1,1], 1.5, 4, target_oce_flux, prior_oce_flux, post1_oce_flux, post2_oce_flux,
          [0, 1.4], "Carbon Flux to Oceans", "PgC/yr 2014–2023", 'Ocean_flux')

dist_plot(ax[1,2], 0, 5, target_land_flux, prior_land_flux, post1_land_flux, post2_land_flux,
          [0, 1], "Carbon Flux to Land", "PgC/yr  2014–2023", 'Land_flux')

dist_plot(ax[2,0], 0, 8, target_ecs, prior_ecs, post1_ecs, post2_ecs,
          [0, 0.5], "ECS", "°C", 'ECS')

dist_plot(ax[2,1], 0, 4, target_tcr, prior_tcr, post1_tcr, post2_tcr,
          [0, 1.5], "TCR", "°C", 'TCR')


legend_elements = [
    Line2D([0], [0], color=colors["target"], lw=2, label="Target"),
    Line2D([0], [0], color=colors["prior"], lw=2, label="Prior"),
    Line2D([0], [0], color=colors["post1"], lw=2, label="Temp+Flux RMSE"),
    Line2D([0], [0], color=colors["post2"], lw=2, label="All constraints"),
]

ax[2,2].legend(handles=legend_elements)

fig.tight_layout()

plt.savefig(
    "../../calibration/plots/constraints.png"
)


#%%

df_obs = pd.read_csv(
    f'{indir}/rcmip_phase3_processed_constraining_data_{rcmip_version}.csv')

df_obs = df_obs.reset_index(drop=True)
years_obs = [col for col in df_obs.columns if str(col).isdigit()]
gmst = df_obs.loc[df_obs["Variable"] == "Global Mean Surface Temperature (GMST)"][years_obs]

gmst_series = gmst[years_obs].iloc[0]
gmst_series.index = gmst_series.index.astype(int)

gmst = gmst_series.to_numpy()
time_temp = gmst_series.index.to_numpy()


temp_hist = df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=2020)].drop(columns='Year').values
temp_hist_offset = temp_hist - temp_pi

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].fill_between(
    np.arange(1850, 2021),
    np.min(temp_hist_offset[:, draws[0].index], axis=1),
    np.max(temp_hist_offset[:, draws[0].index], axis=1),
    color="#000000",
    alpha=0.2,
)
ax[0].fill_between(
    np.arange(1850, 2021),
    np.percentile(temp_hist_offset[:, draws[0].index], 5, axis=1,),
    np.percentile(temp_hist_offset[:, draws[0].index], 95, axis=1,),
    color="#000000",
    alpha=0.2,
)
ax[0].fill_between(
    np.arange(1850, 2021),
    np.percentile(temp_hist_offset[:, draws[0].index], 16, axis=1,),
    np.percentile(temp_hist_offset[:, draws[0].index], 84, axis=1,),
    color="#000000",
    alpha=0.2,
)
ax[0].plot(
    np.arange(1850, 2021),
    np.median(temp_hist_offset[:, draws[0].index], axis=1,),
    color="#000000",
)

ax[0].plot(time_temp, gmst, color="b", label="Observations")

ax[0].legend(frameon=False, loc="upper left")

ax[0].set_xlim(1850, 2025)
ax[0].set_ylim(-1, 5)
ax[0].set_ylabel("°C relative to 1850-1900")
ax[0].axhline(0, color="k", ls=":", lw=0.5)
ax[0].set_title("Temperature anomaly: posterior")



plt.tight_layout()
plt.savefig(
    "../../calibration/plots/final_reweighted_temp.png"
)

#%%

np.savetxt(
    "../../data/constraining/runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)

#%%

draws[0].to_csv(f'../../data/constraining/draws_{output_ensemble_size}.csv')


with open('../../data/constraining/distributions.pickle', 'wb') as handle:
    pickle.dump(dict_distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)

