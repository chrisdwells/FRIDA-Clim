import pandas as pd

start_year = 1750

# FRIDA emissions file from https://github.com/chrisdwells/FRIDA-emissions/tree/main/data/outputs

frida_emissions = pd.read_csv('../data/external/climate_calibration_data.csv')

# Food and Land Use emissions internal to carbon cycle
frida_priors_emissions = frida_emissions.drop('Emissions.CO2 Emissions from Food and Land Use', axis=1)

frida_priors_emissions.to_csv('../data/processed_for_frida/priors_inputs/forcings_frida_iam.csv', index=False)

