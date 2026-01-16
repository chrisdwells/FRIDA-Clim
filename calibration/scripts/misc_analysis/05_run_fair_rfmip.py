import numpy as np
import pandas as pd
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

print("Running SSP scenarios RFMIP...")

output_ensemble_size = 841

scenarios = {
    "ssp119":'a',
    "ssp245":'c',
    "ssp534-over":'x',
    "ssp585":'e',
}

for scenario in scenarios.keys():
    
    # force with RFMIP - use solar specie to get into FaIR here
    df_erf = pd.read_csv(
        "data/input/solar_erf_timebounds.csv", index_col="year"
    )
    
    df_erf = pd.read_csv(
        f'../../data/external/forcing/ssp_forcings/table_A3.4{scenarios[scenario]}_{scenario}_ERF_1750-2500_best_estimate.csv')

    rfmip_forcing = np.zeros(751)
    rfmip_forcing = df_erf['total'].values
    
    df_methane = pd.read_csv(
        "data/input/CH4_lifetime.csv", index_col=0,
    )
    df_configs = pd.read_csv(
        "data/input/calibrated_constrained_parameters.csv", index_col=0,
    )
    df_landuse = pd.read_csv(
        "data/input/landuse_scale_factor.csv", index_col=0,
    )
    df_lapsi = pd.read_csv(
        "data/input/lapsi_scale_factor.csv", index_col=0,
    )
    valid_all = df_configs.index
    
    trend_shape = np.ones(751)
    trend_shape[:271] = np.linspace(0, 1, 271)
    
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2500, 1)
    f.define_scenarios([scenario])
    f.define_configs(valid_all)
    species, properties = read_properties()
    species = ['Solar']
            
    f.define_species(species, properties)
    f.allocate()
    
    da_emissions = xr.load_dataarray(
        "data/input/ssps_harmonized_1750-2499.nc"
    )
    
    da = da_emissions.loc[dict(config="unspecified", scenario=[scenario])]#[:351, ...]
    fe = da.expand_dims(dim=["config"], axis=(2))
    f.emissions = fe.drop("config") * np.ones((1, 1, output_ensemble_size, 1))

    
    fill(
        f.forcing,
        rfmip_forcing[:, None, None],
        specie="Solar",
    )
    
    # climate response
    fill(
        f.climate_configs["ocean_heat_capacity"],
        df_configs.loc[:, "clim_c1":"clim_c3"].values,
    )
    fill(
        f.climate_configs["ocean_heat_transfer"],
        df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values,
    )  # not massively robust, since relies on kappa1, kappa2, kappa3 being in adjacent cols
    fill(
        f.climate_configs["deep_ocean_efficacy"],
        df_configs["clim_epsilon"].values.squeeze(),
    )
    fill(
        f.climate_configs["gamma_autocorrelation"],
        df_configs["clim_gamma"].values.squeeze(),
    )
    fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])
    
    # species level
    f.fill_species_configs()
    
    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    
    f.run()
    
    f.temperature.sel(layer=0, scenario = scenario).quantile((0.05,0.17,0.50,0.83,0.95), dim='config').to_pandas().T.to_csv(
        f'../../data/external/fair_input/RFMIP/temperature_{scenario}.csv')
    

