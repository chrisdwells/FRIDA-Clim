import pandas as pd
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

GtC_to_MtCO2 = 3670

start_year = 0
end_year = 500

expts = ['flat10', 'flat10-zec', 'flat10-cdr', 'flat10-nz']
os.makedirs("../../calibration/data/flat10_output/", exist_ok=True)

for calib in ['frida_iam', 'rcmip']:
    os.makedirs(f'../../calibration/{calib}/data/flat10_output/', exist_ok=True)
    for expt in expts:
        fname = f"../../calibration/{calib}/data/flat10_output/{expt}_output.csv"
        if os.path.isfile(fname) == False:
            df_blank = pd.DataFrame(list())
            df_blank.to_csv(fname)

ems = np.full(end_year - start_year + 1, np.nan)
time = np.arange(start_year, end_year+1)

# they all have 10 to 100

ems[:100] = 10
ems_flat10 = copy.deepcopy(ems)
ems_flat10_zec = copy.deepcopy(ems)
ems_flat10_cdr = copy.deepcopy(ems)
ems_flat10_nz = copy.deepcopy(ems)

# flat10; 10 all the way
ems_flat10[:] = 10

# zec; 0 from 100 on
ems_flat10_zec[100:] = 0

# cdr; linear down to 200, then -10 to 300; then 0
ems_flat10_cdr[100:200] = np.arange(10, -10, -0.2)
ems_flat10_cdr[200:300] = -10
ems_flat10_cdr[300:] = 0

# nz; linear down to 150, then 0
ems_flat10_nz[100:150] = np.arange(10, 0, -0.2)
ems_flat10_nz[150:] = 0

# make pi emissions to check equilibrium
ems_pi = np.full(end_year - start_year + 1, np.nan)
ems_pi[:] = 0


plt.plot(time, ems_flat10, color='grey', label='flat10')
plt.plot(time, ems_flat10_zec, color='C1', label='flat10-zec')
plt.plot(time, ems_flat10_cdr, color='C0', label='flat10-cdr')
plt.plot(time, ems_flat10_nz, color='C2', label='flat10-nz', linestyle='--')

plt.legend()

plt.xlim([0,300])
plt.ylim([-15,15])
plt.ylabel('Emissions [PgC]')
plt.xlabel('Simulation Years')

#%%

df_flat10 = pd.DataFrame()
df_flat10['Year'] = time
df_flat10 = df_flat10.set_index('Year')
df_flat10['Emissions.CO2 Emissions from Fossil use'] = ems_flat10*GtC_to_MtCO2
df_flat10.to_csv('../data/processed_for_frida/emissions_flat10.csv')


df_flat10_zec = pd.DataFrame()
df_flat10_zec['Year'] = time
df_flat10_zec = df_flat10_zec.set_index('Year')
df_flat10_zec['Emissions.CO2 Emissions from Fossil use'] = ems_flat10_zec*GtC_to_MtCO2
df_flat10_zec.to_csv('../data/processed_for_frida/emissions_flat10_zec.csv')


df_flat10_cdr = pd.DataFrame()
df_flat10_cdr['Year'] = time
df_flat10_cdr = df_flat10_cdr.set_index('Year')
df_flat10_cdr['Emissions.CO2 Emissions from Fossil use'] = ems_flat10_cdr*GtC_to_MtCO2
df_flat10_cdr.to_csv('../data/processed_for_frida/emissions_flat10_cdr.csv')


df_flat10_nz = pd.DataFrame()
df_flat10_nz['Year'] = time
df_flat10_nz = df_flat10_nz.set_index('Year')
df_flat10_nz['Emissions.CO2 Emissions from Fossil use'] = ems_flat10_nz*GtC_to_MtCO2
df_flat10_nz.to_csv('../data/processed_for_frida/emissions_flat10_nz.csv')


df_pi = pd.DataFrame()
df_pi['Year'] = time
df_pi = df_pi.set_index('Year')
df_pi['Emissions.CO2 Emissions from Fossil use'] = ems_pi
df_pi.to_csv('../data/processed_for_frida/emissions_pi.csv')

