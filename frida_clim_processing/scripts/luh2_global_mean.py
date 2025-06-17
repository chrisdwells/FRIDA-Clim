import xarray
import numpy as np

luh2_vars= ['primf', 'primn', 'secdf', 'secdn', 'urban', 'c3ann', 
            'c4ann', 'c3per', 'c4per', 'c3nfx', 'pastr', 'range', 
            'secmb', 'secma']

df_in = xarray.open_dataset('states.nc')

for var in luh2_vars:
    ds = df_in[var]

    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    
    ds_weighted = ds.weighted(weights)
    
    ds_weighted_mean = ds_weighted.mean(("lon", "lat"))

    ds_weighted_mean.to_netcdf(f'globave/{var}_globave.nc')


