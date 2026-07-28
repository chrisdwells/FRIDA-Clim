import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import glob
# import math

load_dotenv()

# This processes the data from the RCMIP runs and converts to RCMIP format.
# We save 1 csv by expt now as this is what RCMIP expects - the 100mb file
# limit doesn't seem to apply...

output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

model_version = os.getenv("MODEL_VERSION")

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'

# for the units label - we convert all to RCMIP units 
df_vars = pd.read_csv(f'{indir}/rcmip_phase3_protocol_{rcmip_version}_variable_definitions.csv')

csvs = glob.glob('../../data/frida_clim_output/*.csv')

# default - choose all experiments
# expts = [os.path.splitext(os.path.basename(f))[0] for f in csvs]
expts = [os.path.splitext(os.path.basename(f))[0] for f in csvs if 'esm-1pct-brch-' in os.path.basename(f)]

#%%
# build in exceptions - don't want CO2 or CH4 conc if it's conc-driven for that species
skip_vars = {expt: [] for expt in expts}
for expt in expts:
    if 'allGHG' not in expt:
        skip_vars[expt].append('Atmospheric Concentrations|CH4')
    if 'esm' not in expt and 'methanemip' not in expt:
        skip_vars[expt].append('Carbon Pool|Atmosphere')
        skip_vars[expt].append('Atmospheric Concentrations|CO2')
    if 'esm-1pct-brch' not in expt:
        skip_vars[expt].append('Emissions|CO2')
 
def load_frida(var):
    def process(df):
        var_out = np.full((len(df['Year']), output_ensemble_size), np.nan)
        for i in range(output_ensemble_size):
            colname = f'="Run {i+1}: {var}[1]"'
            var_out[:,i] = df[colname]
        return var_out
    return process


def load_frida_offset(var, y1, y2): # note we don't use this for gmst currently but here if needed; use for ocean temp change.
    def process(df):
        var_out = np.full((len(df['Year']), output_ensemble_size), np.nan)
        for i in range(output_ensemble_size):
            colname = f'="Run {i+1}: {var}[1]"'
            var_out[:,i] = df[colname] - np.mean(df[colname].loc[(y1 <= df['Year']) & (df['Year'] <= y2)])
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

def calc_pH(pH_warm_transform, pH_cold_transform,
            vol_warm_transform, vol_cold_transform):

    def wrapped(df):
        pH_w = pH_warm_transform(df)
        pH_c = pH_cold_transform(df)
        V_w  = vol_warm_transform(df)
        V_c  = vol_cold_transform(df)

        H_w = 10.0 ** (-pH_w)
        H_c = 10.0 ** (-pH_c)

        H_mix = (H_w * V_w + H_c * V_c) / (V_w + V_c)
        pH_mix = -np.log10(H_mix)

        return pH_mix
    return wrapped


def save_df_in_chunks(df, name, chunk_size_rows):
    start = 0
    chunk_id = 0

    while start < len(df):
        end = start + chunk_size_rows
        df_chunk = df.iloc[start:end]
        df_chunk.to_csv(f"../../data/processed_rcmip/{name}_{chunk_id+1}.csv", index=False)
        chunk_id += 1
        start = end
        
GtC_to_MtCO2 = 3.664*1000
MtCO2_per_ppm = 7800.3
YJ_to_ZJ = 1000
HeatUptake_to_OHC = 0.91
warm_ocean_frac = 0.85
m_to_cm = 100

rcmip_from_frida_dict = {
    "Atmospheric Concentrations|CH4": load_frida("CH4 Forcing.Atmospheric CH4 Concentration"),
    # "Atmospheric Concentrations|N2O": load_frida("N2O Forcing.Atmospheric N2O Concentration"),
    
    "Carbon Flux|Land|Decomposition": scale_units(load_frida("soil carbon decay.Total litter to soil carbon"), factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Heterotrophic Respiration|Litter": scale_units(load_frida("soil carbon emissions.litter emissions GtC"), factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Heterotrophic Respiration|Soil": scale_units(load_frida("soil carbon emissions.gross soil carbon emissions GtC"), factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Litterfall|Litter":  scale_units(load_frida("soil carbon decay.Total litter to soil carbon"), factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Litterfall|Soil": scale_units(load_frida("soil carbon emissions.litter emissions GtC"), factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Litterfall": add_many( 
                                scale_units(load_frida("soil carbon decay.Total litter to soil carbon"), factor = GtC_to_MtCO2),
                                scale_units(load_frida("soil carbon emissions.litter emissions GtC"), factor = GtC_to_MtCO2)),
    "Carbon Flux|Land|Net Primary Production":  scale_units(load_frida("Terrestrial Carbon Balance.Terrestrial net primary production"),
                                                            factor = GtC_to_MtCO2),
    "Carbon Flux|Land|Net Primary Production|Soil": add_many(             
                                scale_units(load_frida("soil carbon decay.Total litter to soil carbon"), factor = GtC_to_MtCO2),
                                scale_units(load_frida("Terrestrial Carbon Balance.Annual carbon uptake in peatlands"), factor = GtC_to_MtCO2),
                                ),
    "Carbon Flux|Land|Net Primary Production|Vegetation": add_many( 
                                scale_units(load_frida("Terrestrial Carbon Balance.Terrestrial net primary production"), factor = GtC_to_MtCO2),
                                signed(scale_units(load_frida("soil carbon decay.Total litter to soil carbon"), factor = GtC_to_MtCO2), -1.0),
                                signed(scale_units(load_frida("Terrestrial Carbon Balance.Annual carbon uptake in peatlands"), factor = GtC_to_MtCO2), -1.0),
                                ),
    
    # dont want as no product pool?
    # "Carbon Flux|Land|Product Decomposition": scale_units(load_frida("Forest.timber"), factor = GtC_to_MtCO2),      
    # "Carbon Flux|Land|Product Production": scale_units(load_frida("Forest.timber"), factor = GtC_to_MtCO2), 
    "Carbon Flux|Land|Other": add_many( 
                            scale_units(load_frida("Forest.Forest burned biomass emissions"), factor = GtC_to_MtCO2), 
                            scale_units(load_frida("Grass.animal grazing"), factor = GtC_to_MtCO2), 
                                    ),
    "Carbon Flux|Ocean|Net surface to deep": add_many( 
                    scale_units(load_frida("Ocean.Downward transport of carbon via the overturning circulation"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Convective mixing of carbon between polar surface ocean and deep ocean"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Biological carbon pump C export to the deep ocean"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Biological carbon pump C export from warm surface to intermediate ocean"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Biological carbon pump C export from cold surface to intermediate ocean"), factor = GtC_to_MtCO2),    
                    
                    signed(scale_units(load_frida("Ocean.Mixing of carbon from intermediate depth to the warm surface ocean"), factor = GtC_to_MtCO2), -1.0),    
                    signed(scale_units(load_frida("Ocean.Poleward transport of carbon via the overturning circulation"), factor = GtC_to_MtCO2), -1.0),    
                        ),
    
    "Carbon Flux|Ocean|Net surface to deep|Inorganic": add_many( 
                    scale_units(load_frida("Ocean.Downward transport of carbon via the overturning circulation"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Convective mixing of carbon between polar surface ocean and deep ocean"), factor = GtC_to_MtCO2),    
                    
                    signed(scale_units(load_frida("Ocean.Mixing of carbon from intermediate depth to the warm surface ocean"), factor = GtC_to_MtCO2), -1.0),    
                    signed(scale_units(load_frida("Ocean.Poleward transport of carbon via the overturning circulation"), factor = GtC_to_MtCO2), -1.0),    
                        ),
    
    "Carbon Flux|Ocean|Net surface to deep|Organic": add_many( 
                    scale_units(load_frida("Ocean.Biological carbon pump C export to the deep ocean"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Biological carbon pump C export from warm surface to intermediate ocean"), factor = GtC_to_MtCO2),    
                    scale_units(load_frida("Ocean.Biological carbon pump C export from cold surface to intermediate ocean"), factor = GtC_to_MtCO2),    
                        ),
    
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
    "Carbon Pool|Ocean|Deep": add_many( 
        scale_units(load_frida("Ocean.Intermediate depth ocean carbon reservoir"), factor = GtC_to_MtCO2),
        scale_units(load_frida("Ocean.Deep ocean ocean carbon reservoir"), factor = GtC_to_MtCO2),
            ),
    "Carbon Pool|Ocean|Surface": add_many( 
        scale_units(load_frida("Ocean.Warm surface ocean carbon reservoir"), factor = GtC_to_MtCO2),
        scale_units(load_frida("Ocean.Cold surface ocean carbon reservoir"), factor = GtC_to_MtCO2),
            ),
    
    "Net Flux to Atmosphere|CO2": load_frida("CO2 Forcing.Annual change of atmospheric CO2"),
    "Emissions|CO2|Land Use Change": load_frida("Emissions.CO2 Emissions from Food and Land Use"),
    "Emissions|CO2": load_frida("CO2 Forcing.CO2 Emissions"),

    "Natural Fluxes|CO2|Ocean": signed(scale_units(load_frida("Ocean.Air sea co2 flux"), factor = GtC_to_MtCO2), -1.0),
    "Natural Fluxes|CO2|Land": signed(scale_units(load_frida("Emissions.land carbon sink"), factor = GtC_to_MtCO2),   -1.0),    
    
    "Natural Fluxes|CO2": add_many( 
        signed(scale_units(load_frida("Ocean.Air sea co2 flux"), factor = GtC_to_MtCO2), -1.0),
        signed(scale_units(load_frida("Emissions.land carbon sink"), factor = GtC_to_MtCO2), -1.0),
        ),
    
    "Effective Radiative Forcing": load_frida("Forcing.Total Effective Radiative Forcing"),
    "Effective Radiative Forcing|Anthropogenic": load_frida("Forcing.Anthropogenic Effective Radiative Forcing"),
    "Effective Radiative Forcing|Anthropogenic|CO2":  load_frida("CO2 Forcing.CO2 Effective Radiative Forcing"),

    "Heat Content|Ocean": scale_units(load_frida("Energy Balance Model.ocean heat content change"), factor = YJ_to_ZJ*HeatUptake_to_OHC),
    "Heat Uptake": scale_units(load_frida("Energy Balance Model.ocean heat flow"), factor = YJ_to_ZJ),
    "Heat Uptake|Ocean": scale_units(load_frida("Energy Balance Model.ocean heat flow"), factor = YJ_to_ZJ*HeatUptake_to_OHC),

    "Surface Air Temperature Change": load_frida("Energy Balance Model.Surface Temperature Anomaly"),
    "Surface Ocean Temperature Change": add_many( 
                    scale_units(load_frida_offset("Ocean.Warm surface ocean temperature", 170, 1750), factor = warm_ocean_frac),
                    scale_units(load_frida_offset("Ocean.Cold surface ocean temperature", 1750, 1750), factor = (1 - warm_ocean_frac)),
        ),
    
    "Ocean pH": calc_pH(
        load_frida("Ocean.Warm surface ocean pH"),
        load_frida("Ocean.Cold surface ocean pH"),
        load_frida("Ocean.Volume of warm surface ocean reservoir"),
        load_frida("Ocean.Volume of cold surface ocean reservoir"),
        ),

    "Sea Level Change": scale_units(load_frida("Sea Level.Total global sea level anomaly"), factor = m_to_cm),
    "Sea Level Change|Thermal Expansion": scale_units(load_frida("Sea Level.Sea level anomaly from thermal expansion"), factor = m_to_cm),
    "Sea Level Change|Glaciers": scale_units(load_frida("Sea Level.Sea level anomaly from mountain glaciers"), factor = m_to_cm),
    "Sea Level Change|Greenland": scale_units(load_frida("Sea Level.Sea level anomaly from Greenland Ice Sheet"), factor = m_to_cm),
    "Sea Level Change|Antarctica": scale_units(load_frida("Sea Level.Sea level anomaly from Antarctic Ice Sheet"), factor = m_to_cm),
    "Sea Level Change|Land Water Storage": scale_units(load_frida("Sea Level.Sea level anomaly from LWS"), factor = m_to_cm),

    "Atmospheric Concentrations|CO2": load_frida("CO2 Forcing.Atmospheric CO2 Concentration"),

    "Atmospheric Lifetime|CH4": load_frida("CH4 Forcing.CH4 Lifetime"),
    "Atmospheric Lifetime|N2O": load_frida("N2O Forcing.N2O Lifetime"),
    
    "Effective Radiative Forcing|Anthropogenic|Aerosol": load_frida(
        "Aerosol Forcing.Effective Radiative Forcing from Aerosols"),
    "Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-cloud Interactions": load_frida(
        "Aerosol Forcing.Effective Radiative Forcing from Aerosol Cloud Interactions"),
    "Effective Radiative Forcing|Anthropogenic|Aerosol|Aerosol-radiation Interactions": load_frida(
        "Aerosol Forcing.Effective Radiative Forcing from Aerosol Radiation Interactions"),
    
    "Effective Radiative Forcing|Anthropogenic|Albedo Change":  load_frida("Land Use Forcing.Albedo Forcing"),
    "Effective Radiative Forcing|Anthropogenic|CH4":  load_frida("CH4 Forcing.CH4 Effective Radiative Forcing"),
    "Effective Radiative Forcing|Anthropogenic|F-Gases":  load_frida(
        "Minor GHGs Forcing.HFC134a-eq Effective Radiative Forcing scaled"),
    "Effective Radiative Forcing|Anthropogenic|Montreal Gases":  load_frida(
        "Minor GHGs Forcing.Montreal direct Effective Radiative Forcing scaled"),
    "Effective Radiative Forcing|Anthropogenic|N2O":  load_frida("N2O Forcing.N2O Effective Radiative Forcing"),

    "Effective Radiative Forcing|Anthropogenic|Other|BC on Snow":  load_frida(
        "BC on Snow Forcing.Effective Radiative Forcing of Black Carbon on Snow"),
    "Effective Radiative Forcing|Anthropogenic|Other|Stratospheric H2O":  load_frida(
        "Stratospheric Water Vapour Forcing.Effective Radiative Forcing from the CH4 effect on Stratospheric H2O"),
    "Effective Radiative Forcing|Anthropogenic|Other":  add_many(
        load_frida("BC on Snow Forcing.Effective Radiative Forcing of Black Carbon on Snow"),
        load_frida("Stratospheric Water Vapour Forcing.Effective Radiative Forcing from the CH4 effect on Stratospheric H2O"),
        load_frida("Land Use Forcing.Irrigation forcing"),
        ),
    "Effective Radiative Forcing|Anthropogenic|Ozone": load_frida("Ozone Forcing.Ozone Effective Radiative Forcing"),
}



for expt in expts:
    # if os.path.exists(f"../../data/processed_rcmip/frida_rcmip_output_{expt}.csv"):
    #     continue

    all_rows = []
    df_in = pd.read_csv(f'../../data/frida_clim_output/{expt}.csv')
    years = df_in["Year"].tolist()

    for rcmip_var, process in rcmip_from_frida_dict.items():
        if rcmip_var in skip_vars[expt]:
            print(f'Skipping {rcmip_var} for {expt}')
            continue

        arr = process(df_in)
        
        for i in range(output_ensemble_size):
            row = {
                "climate_model": model_version,
                "model": "undefined for now",
                "scenario": expt,
                "region": "World",
                "variable": rcmip_var,
                "unit": df_vars.loc[
                    df_vars["Variable"] == rcmip_var, "Unit"
                ].iloc[0],
                "ensemble_member": i + 1,
            }

            for y_index, year in enumerate(years):
                row[year] = arr[y_index, i]

            all_rows.append(row)

    df_out = pd.DataFrame(all_rows)
    
    df_out.to_csv(f"../../data/processed_rcmip/frida_rcmip_output_{expt}.csv", index=False)

