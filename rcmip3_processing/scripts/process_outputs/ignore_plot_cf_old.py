import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

expts = ['esm-allGHG-ssp585', 'esm-allGHG-ssp534-over']

def load_frida(var):
    def process(df):
        var_out = np.full((len(df['Year']), output_ensemble_size), np.nan)
        for i in range(output_ensemble_size):
            colname = f'="Run {i+1}: {var}[1]"'
            var_out[:,i] = df[colname]
        return var_out
    return process

def scale_units(transform, factor):
    def wrapped(df):
        return transform(df) * factor
    return wrapped

def add_many(*transforms):
    def wrapped(df):
        import numpy as np
        outs = [t(df) for t in transforms]
        return np.sum(np.stack(outs, axis=0), axis=0)
    return wrapped

def signed(transform, weight):
    def wrapped(df): 
        return transform(df) * weight
    return wrapped

GtC_to_MtCO2 = 3.664*1000
MtCO2_per_ppm = 7800.3
YJ_to_ZJ = 1000
HeatUptake_to_OHC = 0.91
warm_ocean_frac = 0.85
m_to_cm = 100

rcmip_from_frida_dict = {
    "Carbon Flux|Land|Net Primary Production":  scale_units(load_frida("Terrestrial Carbon Balance.Terrestrial net primary production"),
                                                            factor = GtC_to_MtCO2),
    
    "Carbon Pool|Atmosphere": scale_units(load_frida("CO2 Forcing.Atmospheric CO2 Concentration"), factor = MtCO2_per_ppm), 
    
    "Carbon Pool|Land|Soil": scale_units(load_frida("Terrestrial Carbon Balance.Total soil carbon with peat"), factor = GtC_to_MtCO2),    
    "Carbon Pool|Land|Vegetation": scale_units(load_frida("Forest.forest aboveground biomass"), factor = GtC_to_MtCO2),    
    "Carbon Pool|Land": add_many( 
        scale_units(load_frida("Terrestrial Carbon Balance.Total soil carbon with peat"), factor = GtC_to_MtCO2), 
        scale_units(load_frida("Forest.forest aboveground biomass"), factor = GtC_to_MtCO2),
        ),
    
    "Carbon Pool|Ocean": add_many( 
        scale_units(load_frida("Ocean.Warm surface ocean carbon reservoir"), factor = GtC_to_MtCO2),
        scale_units(load_frida("Ocean.Cold surface ocean carbon reservoir"), factor = GtC_to_MtCO2),
        scale_units(load_frida("Ocean.Intermediate depth ocean carbon reservoir"), factor = GtC_to_MtCO2),
        scale_units(load_frida("Ocean.Deep ocean ocean carbon reservoir"), factor = GtC_to_MtCO2),
            ),
    "Natural Fluxes|CO2|Ocean": signed(scale_units(load_frida("Ocean.Air sea co2 flux"), factor = GtC_to_MtCO2), -1.0),
    "Natural Fluxes|CO2|Land": signed(scale_units(load_frida("Emissions.land carbon sink"), factor = GtC_to_MtCO2),   -1.0),    
    
    "Effective Radiative Forcing": load_frida("Forcing.Total Effective Radiative Forcing"),

    "Surface Air Temperature Change": load_frida("Energy Balance Model.Surface Temperature Anomaly"),
    "Effective Radiative Forcing|Anthropogenic|Aerosol": load_frida(
        "Aerosol Forcing.Effective Radiative Forcing from Aerosols"),
}


for expt in expts:
    fig, ax = plt.subplots(4, 3, figsize=(16, 16))
    ax = ax.ravel()
    ax_i = -1

    df_in_new = pd.read_csv(f'../../data/frida_clim_output/{expt}.csv')
    df_in_old = pd.read_csv(f'../../data/frida_clim_output/old_npp/{expt}.csv')

    years = df_in_new["Year"].tolist()
    
    for rcmip_var, process in rcmip_from_frida_dict.items():
        ax_i += 1
        arr_new = process(df_in_new)
        arr_old = process(df_in_old)
        
                
        ax[ax_i].plot(years, np.percentile(arr_new, 50, axis=1), color='C0', label=f'New {expt}')
        ax[ax_i].plot(years, np.percentile(arr_old, 50, axis=1), color='C1', label=f'Old {expt}')

        ax[ax_i].fill_between(years, 
                          np.percentile(arr_new, 5, axis=1), 
                          np.percentile(arr_new, 95, axis=1), 
                          color='C0', linewidth=0, alpha=0.3)
        ax[ax_i].fill_between(years, 
                          np.percentile(arr_old, 5, axis=1), 
                          np.percentile(arr_old, 95, axis=1), 
                          color='C1', linewidth=0, alpha=0.3)

        ax[ax_i].set_title(f'{rcmip_var}')

    ax[0].legend()

    plt.tight_layout()
    plt.savefig(
        f"../../data/frida_clim_output/old_npp/{expt}.png"
    )
