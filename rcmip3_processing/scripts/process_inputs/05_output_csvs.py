import os
import pandas as pd

expts = ['esm-allGHG-hist']

os.makedirs("../../data/frida_clim_output/", exist_ok=True)

output_tables = ['emissions-driven', 'non-idealised_emissions-driven',
               'non-idealised', 'always']

for exp in expts:
    os.makedirs(f"../../data/frida_clim_output/{exp}/", exist_ok=True)
    for tab in output_tables:
        csv = f'../../data/frida_clim_output/{exp}/{tab}.csv'
        if os.path.isfile(csv) == False:
            df_blank = pd.DataFrame(list())
            df_blank.to_csv(csv)

