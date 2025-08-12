import pooch
import pandas as pd
import numpy as np
import os

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

# makes the SSP emissions timeseries for doing the SSP projections - the 
# emissions used in the actual calibration are those used in FRIDA, collated
# at https://github.com/chrisdwells/FRIDA-emissions

# HFC emissions are back-calculated from concentrations

start_year = 1750
end_year = 2500
n_years = end_year - start_year + 1

ssps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

rcmip_emissions_file = pooch.retrieve(
    url=(
        "https://zenodo.org/records/4589756/files/"
        "rcmip-emissions-annual-means-v5-1-0.csv"
    ),
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

df_emis = pd.read_csv(rcmip_emissions_file)

#%%

# HFCs

f = FAIR()

f.define_time(start_year, end_year, 1)


f.define_scenarios(ssps)

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

f_gases = ['CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12',
       'C6F14', 'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2', 'HFC-125',
       'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23',
       'HFC-236fa', 'HFC-245fa', 'HFC-32', 'HFC-365mfc', 'HFC-4310mee']

# source: Hodnebrog et al 2020 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019RG000691
radeff = {
    'HFC-125':      0.23378,
    'HFC-134a':     0.16714,
    'HFC-143a':     0.168,
    'HFC-152a':     0.10174,
    'HFC-227ea':    0.27325,
    'HFC-23':       0.19111,
    'HFC-236fa':    0.25069,
    'HFC-245fa':    0.24498,
    'HFC-32':       0.11144,
    'HFC-365mfc':   0.22813,
    'HFC-4310mee': 0.35731,
    'NF3':          0.20448,
    'C2F6':         0.26105,
    'C3F8':         0.26999,
    'C4F10':      0.36874,
    'C5F12':      0.4076,
    'C6F14':      0.44888,
    'C6F14':      0.44888,
    'C7F16':        0.50312,
    'C8F18':        0.55787,
    'CF4':          0.09859,
    'c-C4F8':       0.31392,
    'SF6':          0.56657,
    'SO2F2':        0.21074,
    'CCl4':         0.16616,
    'CFC-11':       0.25941,
    'CFC-112':      0.28192,
    'CFC-112a':     0.24564,
    'CFC-113':      0.30142,
    'CFC-113a':     0.24094, 
    'CFC-114':      0.31433,
    'CFC-114a':     0.29747,
    'CFC-115':      0.24625,
    'CFC-12':       0.31998,
    'CFC-13':       0.27752,
    'CH2Cl2':       0.02882,
    'CH3Br':        0.00432,
    'CH3CCl3':      0.06454,
    'CH3Cl':        0.00466,
    'CHCl3':        0.07357,
    'HCFC-124':     0.20721,
    'HCFC-133a':    0.14995,
    'HCFC-141b':    0.16065,
    'HCFC-142b':    0.19329,
    'HCFC-22':      0.21385,
    'HCFC-31':      0.068,
    'Halon-1202':   0,       # not in dataset
    'Halon-1211':   0.30014,
    'Halon-1301':   0.29943,
    'Halon-2402':   0.31169,
    'CO2':          0,       # different relationship
    'CH4':          0,       # different relationship
    'N2O':          0        # different relationship
}

# back calculate emissions
lifetime = 14
decay_rate = 1 / lifetime
decay_factor = np.exp(-decay_rate)

mass_atmosphere = 5.1352e18 # kg
molecular_weight_air = 28.97 # g/mol
molecular_weight_hfc134a = 102.03 # g/mol

concentration_per_emission = 1 / (
    mass_atmosphere / 1e18 * molecular_weight_hfc134a / molecular_weight_air
)

#%%

species = {
    'CO2':['Emissions.CO2 Emissions from Fossil use', 'Emissions|CO2|MAGICC Fossil and Industrial'],
    'CH4':['Emissions.Total CH4 Emissions', 'Emissions|CH4'],
    'SO2':['Emissions.Total SO2 Emissions', 'Emissions|Sulfur'],
    'N2O':['Emissions.Total N2O Emissions', 'Emissions|N2O'],
    'N2O non AFOLU':['Emissions.N2O non AFOLU Emissions', 'Emissions|N2O|MAGICC Fossil and Industrial'],
    }

baseline_species = ['CH4', 'SO2', 'N2O', 'N2O non AFOLU']

baseline_only_species = {
    'VOC':['Emissions.Baseline VOC Emissions', 'Emissions|VOC'],
    'CO':['Emissions.Baseline VOC Emissions', 'Emissions|CO'],
    }

for scen in ssps:
    
    df_ssp = pd.DataFrame()
    df_ssp['Year'] = np.arange(start_year, end_year+1)
    df_ssp = df_ssp.set_index('Year')

    for specie in species.keys():
        frida_name = species[specie][0]
        rcmip_name = species[specie][1]
    
        df_ssp[frida_name] = (
            df_emis.loc[
                (df_emis["Scenario"] == scen)
                & (
                    df_emis["Variable"] == rcmip_name
                )
                & (df_emis["Region"] == "World"),
                "1750":"2500",
            ]
            .interpolate(axis=1)
            .values.squeeze()
        )
        
        
    hfc134a_eq = np.zeros(n_years)
    for gas in f_gases:
        hfc134a_eq = hfc134a_eq + f.concentration[:,f.scenarios.index(scen),
                  0,f.species.index(gas)] * radeff[gas] / radeff['HFC-134a']
        
        
    hfc134a_eq_minus_baseline = hfc134a_eq.values - hfc134a_eq.values[0]
    
    new_ems = np.zeros(n_years)
    for i in range(1, n_years):
        new_ems[i] = (hfc134a_eq_minus_baseline[i] - hfc134a_eq_minus_baseline[i-1
                           ]*decay_factor)/concentration_per_emission
      
    df_ssp['Emissions.HFC134a eq Emissions'] = new_ems
        
    df_ssp.to_csv(f'../data/processed_for_frida/emissions_{scen}.csv')

#%%
df_baseline = pd.DataFrame()
df_baseline['Year'] = np.arange(start_year, start_year+1)
df_baseline = df_baseline.set_index('Year')

for specie in baseline_species:
    df_baseline[f'Emissions.{specie} Baseline Emissions'] = df_ssp[species[specie][0]].loc[df_ssp.index == 1750].values[0]
 
# fix NOx baseline separately (as per FaIR)

gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]

baseline_nox = (df_emis.loc[
        (df_emis["Scenario"] == scen)
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"].isin(gfed_sectors)),
        "1750",
    ].sum()* 46.006/ 30.006
    + df_emis.loc[
        (df_emis["Scenario"] == scen)
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
        "1750",
    ].values[0]

    + df_emis.loc[
        (df_emis["Scenario"] == scen)
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
        "1750",
    ].values[0]

)

baseline_nox_non_afolu = df_emis.loc[
    (df_emis["Scenario"] == scen)
    & (df_emis["Region"] == "World")
    & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
    "1750",
].values[0]

df_baseline['Emissions.Baseline NOx non AFOLU Emissions'] = baseline_nox_non_afolu
df_baseline['Emissions.Baseline NOx AFOLU Emissions'] = baseline_nox - baseline_nox_non_afolu
        
df_baseline.to_csv('../data/processed_for_frida/emissions_baseline.csv')

#%%

output_list = ['land', 'aerosols', 'CO2', 'ocean_CO2_flux', 'ocean_heat_content',
               'temperature', 'SLR', 'NPP']

needed_csvs = []

for scen in ssps:
    for output in output_list:
        needed_csvs.append(f'../../calibration/data/ssps_output/{scen}_{output}.csv')

for csv in needed_csvs:
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

