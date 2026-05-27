import os
import pandas as pd

expts = [
    "piControl", "esm-piControl", "esm-allGHG-piControl", "1pctCO2", "1pctCO2-4xext",
    "1pctCO2-cdr", "1pctCO2-bgc", "1pctCO2-rad", "esm-1pct-brch-1000PgC",
    "esm-1pct-brch-2000PgC", "esm-1pct-brch-750PgC", "abrupt-4xCO2", "abrupt-2xCO2",
    "abrupt-0p5xCO2", "esm-pi-cdr-pulse", "esm-pi-CO2pulse", "esm-bell-1000PgC",
    "esm-bell-2000PgC", "esm-bell-750PgC", "historical", "historical-cmip6", "hist-aer",
    "hist-GHG", "hist-CO2", "ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460",
    "ssp534-over", "ssp585", "esm-hist", "esm-hist-cmip6", "esm-ssp119", "esm-ssp126",
    "esm-ssp245", "esm-ssp370", "esm-ssp434", "esm-ssp460", "esm-ssp534-over",
    "esm-ssp585", "esm-allGHG-hist", "esm-allGHG-hist-cmip6", "esm-allGHG-ssp119",
    "esm-allGHG-ssp126", "esm-allGHG-ssp245", "esm-allGHG-ssp370",
    "esm-allGHG-ssp370-lowNTCF", "esm-allGHG-ssp370-lowCH4",
    "esm-allGHG-ssp370-lowNTCF-HighCH4", "esm-allGHG-ssp434", "esm-allGHG-ssp460",
    "esm-allGHG-ssp534-over", "esm-allGHG-ssp534-over-highCH4", "esm-allGHG-ssp585",
    "esm-allGHG-ssp585-lowCH4", "esm-scen7-H", "esm-scen7-HL", "esm-scen7-M",
    "esm-scen7-ML", "esm-scen7-L", "esm-scen7-VL", "esm-scen7-LN", "scen7-HC",
    "scen7-HLC", "scen7-MC", "scen7-MLC", "scen7-LC", "scen7-VLC", "scen7-LNC",
    "esm-allGHG-scen7-H", "esm-allGHG-scen7-HL", "esm-allGHG-scen7-H-CH4L",
    "esm-allGHG-scen7-M", "esm-allGHG-scen7-ML", "esm-allGHG-scen7-L",
    "esm-allGHG-scen7-L-CH4H", "esm-allGHG-scen7-VL", "esm-allGHG-scen7-LN",
    "esm-flat10", "esm-flat10-zec", "esm-flat10-cdr", "esm-flat10-nz", "esm-flat10-rev",
    "esm-flat7.5", "esm-flat7.5-cdr", "esm-flat7.5-zec", "esm-flat7.5-nz",
    "esm-flat7.5-rev", "esm-flat20", "esm-flat20-cdr", "esm-flat20-zec", "esm-flat20-nz",
    "esm-flat20-rev", "methanemip-TM-allGHG", "methanemip-TM+BC-allGHG"
         ]

os.makedirs("../../data/frida_clim_output/", exist_ok=True)

for exp in expts:
    csv = f'../../data/frida_clim_output/{exp}.csv'
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

