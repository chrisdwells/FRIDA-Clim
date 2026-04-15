import pandas as pd
import numpy as np
import copy

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
import os
from dotenv import load_dotenv
load_dotenv()

# makes the csvs with anthropogenic forcings for hist and the ssps - emissions,
# including HFCs via HFC134a-eq, plus Montreal gas ERF and effect on EESC.



rcmip_version = os.getenv("RCMIP_VERSION")
rcmip_version_folder = rcmip_version.replace(".", "_").upper()

indir = f'../../../RCMIP3_protocol_bundle_{rcmip_version_folder}/RCMIP3_input_datafiles'
outdir = '../../data/processed_for_frida'

data_in = pd.read_csv(f'{indir}/rcmip_phase3_emissions_{rcmip_version}.csv')
conc_in = pd.read_csv(f'{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv')

start_year = 1750
end_year = 2500
n_years = end_year - start_year + 1

hist_end_year = 2023
n_years_hist = hist_end_year - start_year + 1

cmip6_hist_end_year = 2015
n_years_cmip6_hist = cmip6_hist_end_year - start_year + 1

ssps = [#'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 
        # 'ssp460', 'ssp534-over', 'ssp585', 
        # 'esm-allGHG-ssp370-lowNTCF', 'esm-allGHG-ssp370-lowCH4', 
        # 'esm-allGHG-ssp370-lowNTCF-HighCH4',
        # 'methanemip-TM-allGHG', 'methanemip-TM+BC-allGHG',
        # 'esm-allGHG-ssp534-over-highCH4',
        'esm-allGHG-ssp585-lowCH4',
        ]

# Make the HFC emissions back-calculated from concentrations
# Note this is a bit hacky - you have to use a modified version of fill_from.py (in utils/)

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

def process_df(df_in, years):
    ems_long = df_in.melt(
        id_vars='Variable',
        value_vars=years,
        var_name='Year',
        value_name='Value'
    )
    
    ems_long['Year'] = ems_long['Year'].astype(int)
    
    ems_out = (
        ems_long
        .pivot(index='Year', columns='Variable', values='Value')
        .sort_index()
    )
    return ems_out

spec_remove = ['Halon-1202', 'NOx aviation']

scen_specific_specs_to_remove = {
    'hist-GHG':['Solar', 'Volcanic'],
    'hist-aer':['Solar', 'Volcanic', 'CH2Cl2', 'CHCl3', 'HFC-152a',
                'HFC-236fa', 'HFC-365mfc', 'NF3', 'C3F8', 'C4F10', 'C5F12',
                'C7F16', 'C8F18', 'c-C4F8', 'SO2F2'],
    'hist-CO2':['Solar', 'Volcanic', 'CH2Cl2', 'CHCl3', 'HFC-152a',
                'HFC-236fa', 'HFC-365mfc', 'NF3', 'C3F8', 'C4F10', 'C5F12',
                'C7F16', 'C8F18', 'c-C4F8', 'SO2F2'],
    'esm-allGHG-ssp370-lowNTCF':['Solar', 'Volcanic'],
    'esm-allGHG-ssp370-lowCH4':['Solar', 'Volcanic'],
    'esm-allGHG-ssp370-lowNTCF-HighCH4':['Solar', 'Volcanic'],
    'esm-allGHG-ssp534-over-highCH4':['Solar', 'Volcanic'],
    'esm-allGHG-ssp585-lowCH4':['Solar', 'Volcanic'],
    }

for scen in ssps:
        
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    
    f.define_scenarios([scen])
    configs = ['test']
    f.define_configs(configs)
    
    species, properties = read_properties()
    
    species = [s for s in species if s not in spec_remove]
    if scen in scen_specific_specs_to_remove.keys():
        species = [s for s in species if s not in scen_specific_specs_to_remove[scen]]
        
    for s in spec_remove:
        properties.pop(s, None)
    
    f.define_species(species, properties)
    
    f.allocate()
    f.fill_species_configs()
    
    f.fill_from_rcmip(emissions_file=f'{indir}/rcmip_phase3_emissions_{rcmip_version}.csv',
                      concentration_file=f'{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv',
                      forcing_file=f'{indir}/rcmip_phase3_forcing_{rcmip_version}.csv')
    
    
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
    
    
    # get Montreal gas effect on EESC concentration and direct ERF
    
    eesc = (
        f.concentration
        .sel(
            config="test",
            specie="Equivalent effective stratospheric chlorine"
        )
    )
    
    
    species_list = f.species_configs.sel(config="test").specie.where(
        (f.species_configs.sel(config="test").cl_atoms > 0) |
        (f.species_configs.sel(config="test").br_atoms > 0),
        drop=True
    ).values.tolist()
    
    montreal_erf = f.forcing.sel(config="test", specie=species_list).sum(dim="specie")
    
    
    
    # process the other emissions and combine into csvs
    
    
    ems_species = {
        'Emissions|CH4': 'Emissions.Total CH4 Emissions',
        'Emissions|CO': 'Emissions.CO Emissions',
        'Emissions|CO2|Energy and Industrial Processes': 'Emissions.CO2 Emissions from Fossil use',
        'Emissions|N2O': 'Emissions.Total N2O Emissions',
        'Emissions|NOx': 'Emissions.Total NOx Emissions',
        'Emissions|Sulfur': 'Emissions.Total SO2 Emissions',
        'Emissions|VOC': 'Emissions.VOC Emissions',
    }
    
    
    
    df_ssp = pd.DataFrame()
    df_ssp['Year'] = np.arange(start_year, end_year+1)
    df_ssp = df_ssp.set_index('Year')
    
    # hfcs        
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
    
    # Montreal
    df_ssp['Ozone Forcing.Montreal gases equivalent effective stratospheric chlorine'
           ] = eesc.sel(scenario=scen).values
    df_ssp['Minor GHGs Forcing.Montreal Gases Effective Radiative Forcing'
           ] = montreal_erf.sel(scenario=scen).values
        
    # Others
    
    ems_filtered = data_in[
        (data_in['Scenario'] == scen) &
        (data_in['Region'] == 'World') &
        (data_in['Variable'].isin(ems_species.keys()))
    ]
        

    year_cols = [c for c in ems_filtered.columns if c.isdigit()]
    
    ems_df = process_df(ems_filtered, year_cols)
    
    df_ssp = df_ssp.join(ems_df)
    df_ssp = df_ssp.rename(columns=ems_species)

    # (as we need to use the full name to get the rcmip data...)
    if 'esm-allGHG-' in scen:
        scen = scen.replace("esm-allGHG-", "") 

    # deal with methanemip - only allGHG
    if 'methanemip' in scen:
        df_ssp.to_csv(f'{outdir}/{scen}.csv')
        continue

    # output full ems-driven
    df_ssp.to_csv(f'{outdir}/esm-allGHG-{scen}.csv')
    
    if 'ssp370-low' in scen or 'ssp534-over-highCH4' in scen or 'ssp585-lowCH4' in scen:
        continue
    
    # make with non-CO2 conc (ie CH4, N2O) for esm-ssp (except the ssp370 variants which are only allGHG)
    df_ssp_nonCO2_conc = copy.deepcopy(df_ssp)
    
    df_ssp_nonCO2_conc = df_ssp_nonCO2_conc.drop(columns=[
        'Emissions.Total CH4 Emissions',
        'Emissions.Total N2O Emissions',
        ])
    df_n2o = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|N2O')
    ]
    df_ssp_nonCO2_conc['N2O Forcing.Atmos N2O exogenous'] = process_df(df_n2o, year_cols)
    
    df_ch4 = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|CH4')
    ]
    df_ssp_nonCO2_conc['CH4 Forcing.Atmos CH4 exogenous'] = process_df(df_ch4, year_cols)
    
    df_ssp_nonCO2_conc.to_csv(f'{outdir}/esm-{scen}.csv')

    
    # and make with CO2, CH4, N2O conc for ssp ie conc-driven
    df_ssp_ghg_conc = copy.deepcopy(df_ssp_nonCO2_conc)
    
    df_ssp_ghg_conc = df_ssp_ghg_conc.drop(columns=[
        'Emissions.CO2 Emissions from Fossil use',
        ])
    
    df_co2 = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|CO2')
    ]
    
    df_ssp_ghg_conc['CO2 Forcing.Atmos CO2 exogenous'] = process_df(df_co2, year_cols)
    
    df_ssp_ghg_conc.to_csv(f'{outdir}/{scen}.csv')

#%%
for scen in ['historical', 'historical-cmip6', 'hist-GHG', 'hist-aer', 'hist-CO2']:
    
    end_year = hist_end_year
    hist_n_years = n_years_hist
    if 'cmip6' in scen:
        end_year = cmip6_hist_end_year
        hist_n_years = n_years_cmip6_hist
    
    # historical - inputs slightly different to ssps so can't just crop them..
    
    # HFCs
    
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    
    f.define_scenarios([scen])
    configs = ['test']
    f.define_configs(configs)
    
    species, properties = read_properties()
    
    species = [s for s in species if s not in spec_remove]
    if scen in scen_specific_specs_to_remove.keys():
        species = [s for s in species if s not in scen_specific_specs_to_remove[scen]]
        
    for s in spec_remove:
        properties.pop(s, None)
    
    f.define_species(species, properties)
    
    f.allocate()
    f.fill_species_configs()
    
    f.fill_from_rcmip(emissions_file=f'{indir}/rcmip_phase3_emissions_{rcmip_version}.csv',
                      concentration_file=f'{indir}/rcmip_phase3_concentrations_{rcmip_version}.csv',
                      forcing_file=f'{indir}/rcmip_phase3_forcing_{rcmip_version}.csv')
    
    
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
    
    
    eesc = (
        f.concentration
        .sel(
            config="test",
            specie="Equivalent effective stratospheric chlorine"
        )
    )
    
    
    species_list = f.species_configs.sel(config="test").specie.where(
        (f.species_configs.sel(config="test").cl_atoms > 0) |
        (f.species_configs.sel(config="test").br_atoms > 0),
        drop=True
    ).values.tolist()
    
    montreal_erf = f.forcing.sel(config="test", specie=species_list).sum(dim="specie")
    
    
    df_hist = pd.DataFrame()
    df_hist['Year'] = np.arange(start_year, end_year+1)
    df_hist = df_hist.set_index('Year')
    
    # exclude these for hist-aer, co2 - set to the 1750 value of the previous ssp
    # (this will probably fail if no ssps selected..)
    if scen in ['hist-aer', 'hist-CO2']:
        df_hist['Emissions.HFC134a eq Emissions'] = np.full(hist_n_years, 
                    df_ssp['Emissions.HFC134a eq Emissions'].loc[df_ssp.index == 1750].values[0])
        df_hist['Ozone Forcing.Montreal gases equivalent effective stratospheric chlorine'] = np.full(hist_n_years, 
                    df_ssp['Ozone Forcing.Montreal gases equivalent effective stratospheric chlorine'].loc[df_ssp.index == 1750].values[0])
        df_hist['Minor GHGs Forcing.Montreal Gases Effective Radiative Forcing'] = np.full(hist_n_years, 
                    df_ssp['Minor GHGs Forcing.Montreal Gases Effective Radiative Forcing'].loc[df_ssp.index == 1750].values[0])

    else:
        # hfcs        
        hfc134a_eq = np.zeros(hist_n_years)
        for gas in f_gases:
            hfc134a_eq = hfc134a_eq + f.concentration[:,f.scenarios.index(scen),
                      0,f.species.index(gas)] * radeff[gas] / radeff['HFC-134a']
            
        hfc134a_eq_minus_baseline = hfc134a_eq.values - hfc134a_eq.values[0]
        
        new_ems = np.zeros(hist_n_years)
        for i in range(1, hist_n_years):
            new_ems[i] = (hfc134a_eq_minus_baseline[i] - hfc134a_eq_minus_baseline[i-1
                               ]*decay_factor)/concentration_per_emission
          
        df_hist['Emissions.HFC134a eq Emissions'] = new_ems
        
        # Montreal
        df_hist['Ozone Forcing.Montreal gases equivalent effective stratospheric chlorine'
               ] = eesc.sel(scenario=scen).values
        df_hist['Minor GHGs Forcing.Montreal Gases Effective Radiative Forcing'
               ] = montreal_erf.sel(scenario=scen).values
            
    # Others
    
    ems_filtered = data_in[
        (data_in['Scenario'] == scen) &
        (data_in['Region'] == 'World') &
        (data_in['Variable'].isin(ems_species.keys()))
    ]
        
    
    year_cols = [c for c in ems_filtered.columns if c.isdigit()]
    
    ems_df = process_df(ems_filtered, year_cols)
    
    df_hist = df_hist.join(ems_df)
    df_hist = df_hist.rename(columns=ems_species)
        

    # make conc-driven first as hist-aer,ghg,co2 only have this
    df_ghg_conc = copy.deepcopy(df_hist)
    
    df_ghg_conc = df_ghg_conc.drop(columns=[
        'Emissions.CO2 Emissions from Fossil use',
        'Emissions.Total CH4 Emissions',
        'Emissions.Total N2O Emissions',
        ])
    
    df_n2o = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|N2O')
    ]
    df_ghg_conc['N2O Forcing.Atmos N2O exogenous'] = process_df(df_n2o, year_cols)
    
    df_ch4 = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|CH4')
    ]
    df_ghg_conc['CH4 Forcing.Atmos CH4 exogenous'] = process_df(df_ch4, year_cols)
    
    df_co2 = conc_in[
        (conc_in['Scenario'] == scen) &
        (conc_in['Region'] == 'World') &
        (conc_in['Variable'] == 'Atmospheric Concentrations|CO2')
    ]
    
    df_ghg_conc['CO2 Forcing.Atmos CO2 exogenous'] = process_df(df_co2, year_cols)
    
    df_ghg_conc.to_csv(f'{outdir}/{scen}.csv')
    
    if scen in ['hist-GHG', 'hist-aer', 'hist-CO2']:
        continue
    
    scen_name = scen.replace('historical', 'hist') # ems-driven historical are esm-hist...
    
    # output esm-allGHG
    df_hist.to_csv(f'{outdir}/esm-allGHG-{scen_name}.csv')

    # and process co2 ems version
    df_nonCO2_conc = copy.deepcopy(df_hist)
    df_nonCO2_conc['N2O Forcing.Atmos N2O exogenous'] = process_df(df_n2o, year_cols)
    df_nonCO2_conc['CH4 Forcing.Atmos CH4 exogenous'] = process_df(df_ch4, year_cols)

    df_nonCO2_conc.to_csv(f'{outdir}/esm-{scen_name}.csv')

    