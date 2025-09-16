import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Process outputs into format to go into FRIDA
# Run FRIDA-Clim_posteriors, then script 12, to check posteriors run OK.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
calibration = os.getenv("CALIBRATION")

runids = np.loadtxt(
    f"../{calibration}/data/constraining/runids_rmse_reweighted_pass.csv",
).astype(np.int64)

#%%

df_prior_params = pd.read_csv(f"../data/priors_input/priors_inputs_{samples}.csv")


df_posterior_params = pd.DataFrame(data=df_prior_params.values[runids], columns=df_prior_params.keys())

df_posterior_params.to_csv(
    f"../{calibration}/data/constraining/frida_clim_inputs_{output_ensemble_size}_from_{samples}_1750_inits.csv",
    index=False,
)
