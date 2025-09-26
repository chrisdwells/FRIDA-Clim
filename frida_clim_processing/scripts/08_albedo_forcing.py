import pandas as pd
import numpy as np

# need to get scaling factor from albedo to forcing, to match pd estimates.
# use albedo data from an ensemble of FRIDA-Clim which used exog albedo forcing

df_albedo = pd.read_csv('../data/external/frida_albedo.csv', index_col=0)
nyears = 273
output_ensemble_size = 100

def loaddata(df, n_years, members, varname, offset=False):
    var_data = np.full((n_years, members), np.nan)
    for i in np.arange(members):
        var_data[:,i] = df[f'="Run {i+1}: {varname}"'][:n_years]
    if offset == True:
        var_data = var_data - var_data[0,:]
    return var_data


albedo_timeseries =  np.median(loaddata(df_albedo, nyears, output_ensemble_size, "Land Use and Agriculture.Albedo[1]"), axis=1)

# pi to present day value (2019) scaled to give -0.15 W/m2 as IPCC AR6 WG1 Ch7

albedo_scaling = -0.15/(albedo_timeseries[269] - albedo_timeseries[0])

# this albedo_scaling becomes Land Use Forcing per change in Land Use albedo