import pandas as pd
import os

from dotenv import load_dotenv


load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))



vars_to_drop = [
'Aerosol Forcing.Scaling Aerosol Cloud Interactions Effective Radiative Forcing scaling factor[1]',
'Aerosol Forcing.Logarithmic Aerosol Cloud Interactions Effective Radiative Forcing scaling factor[1]',
'Aerosol Forcing.Effective Radiative Forcing from Aerosol Radiation Interactions per unit SO2 Emissions[1]',
'CH4 Forcing.Calibration scaling of CH4 forcing[1]',
'N2O Forcing.Calibration scaling of N2O forcing[1]',
'Minor GHGs Forcing.Calibration scaling of Minor GHG forcing[1]',
'Stratospheric Water Vapour Forcing.Calibration scaling of Stratospheric H2O forcing[1]',
'BC on Snow Forcing.Calibration scaling of Black Carbon on Snow forcing[1]',
'Land Use Forcing.Calibration scaling of Albedo forcing[1]',
'Land Use Forcing.Calibration scaling of Irrigation forcing[1]',
'Natural Forcing.Calibration scaling of Volcano forcing[1]',
'CO2 Forcing.Calibration scaling of CO2 forcing[1]',
'Natural Forcing.Amplitude of Effective Radiative Forcing from Solar Output Variations[1]',
'Natural Forcing.Linear trend in Effective Radiative Forcing from Solar Output Variations[1]',
'Ozone Forcing.Ozone forcing per unit CH4 concentration change[1]',
'Ozone Forcing.Ozone forcing per unit N2O concentration change[1]',
'Ozone Forcing.Ozone forcing per unit Montreal gases equivalent effective stratospheric chlorine concentration change[1]',
'Ozone Forcing.Ozone forcing per unit CO emissions change[1]',
'Ozone Forcing.Ozone forcing per unit VOC emissions change[1]',
'Ozone Forcing.Ozone forcing per unit NOx emissions change[1]',
        ]

df_posterior_params = pd.read_csv(
    f"../data/constraining/frida_clim_inputs_{output_ensemble_size}_from_{samples}_1750_inits.csv",
)

df_posterior_params_dropped = df_posterior_params.drop(columns=vars_to_drop)

df_posterior_params_dropped.to_csv(
    f"../data/constraining/frida_clim_inputs_{output_ensemble_size}_from_{samples}_1750_inits_flat10.csv",
    index=False,
)

