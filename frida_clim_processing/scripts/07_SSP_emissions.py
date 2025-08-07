import pooch
import pandas as pd
import numpy as np

# makes the SSP emissions timeseries for doing the SSP projections - the 
# emissions used in the actual calibration are those used in FRIDA, collated
# at https://github.com/chrisdwells/FRIDA-emissions

start_year = 1750
end_year = 2500

ssps = ['ssp119', 'ssp126ext', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534ext', 'ssp585ext']

rcmip_emissions_file = pooch.retrieve(
    url=(
        "https://zenodo.org/records/4589756/files/"
        "rcmip-emissions-annual-means-v5-1-0.csv"
    ),
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

df_emis = pd.read_csv(rcmip_emissions_file)

#%%

species = {
    'CO2':['Emissions.CO2 Emissions from Fossil use', 'Emissions|CO2|MAGICC Fossil and Industrial'],
    'CH4':['Emissions.Total CH4 Emissions', 'Emissions|CH4'],
    'SO2':['Emissons.Total SO2 Emissions', 'Emissions|Sulfur'],
    'N2O':['Emissions.Total N2O Emissions', 'Emissions|N2O'],
    'N2O non AFOLU':['Emissions.N2O non AFOLU Emissions', 'Emissions|N2O|MAGICC Fossil and Industrial'],
    }

baselines_for_species = {
    'CH4':'Emissions.CH4 Baseline Emissions',
    'SO2':'Emissions.SO2 Baseline Emissions',
    'N2O':'Emissions.N2O Baseline Emissions',
    'N2O non AFOLU':'Emissions.Baseline N2O non AFOLU Emissions',
    }

baseline_only_species = {
    'VOC':['Emissions.Baseline VOC Emissions', 'Emissions|VOC'],
    'CO':['Emissions.Baseline VOC Emissions', 'Emissions|CO'],
    }

for scen in ssps:
    
    df_ssp = pd.DataFrame()
    df_ssp['Year'] = np.arange(start_year, end_year+1)
    df_ssp = df_ssp.set_index('Year')
    
    df_ssp_baseline = pd.DataFrame()
    

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
        
        if specie in baselines_for_species.keys():
            
            df_ssp_baseline[baselines_for_species[specie]] = df_emis.loc[
                (df_emis["Scenario"] == scen)
                & (
                    df_emis["Variable"] == rcmip_name
                )
                & (df_emis["Region"] == "World"),
                "1750",
            ]
    
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
    
    df_ssp_baseline['Emissions.Baseline NOx non AFOLU Emissions'] = baseline_nox_non_afolu
    df_ssp_baseline['Emissions.Baseline NOx AFOLU Emissions'] = baseline_nox - baseline_nox_non_afolu

            
    erg

#%%

for var in df_emis["Variable"]:
    if 'N2O' in var:
        print(var)
