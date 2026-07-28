import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = {
    "1pctCO2":
        "../../data/processed_rcmip/frida_rcmip_output_1pctCO2.csv",
    "Branch 1000 PgC":
        "../../data/processed_rcmip/frida_rcmip_output_esm-1pct-brch-1000PgC.csv",
    "Branch 2000 PgC":
        "../../data/processed_rcmip/frida_rcmip_output_esm-1pct-brch-2000PgC.csv",
    "Branch 750 PgC":
        "../../data/processed_rcmip/frida_rcmip_output_esm-1pct-brch-750PgC.csv",
    "esm-flat10-zec":
        "../../data/processed_rcmip/frida_rcmip_output_esm-flat10-zec.csv",
}

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "1pctCO2": "tab:blue",
    "Branch 1000 PgC": "tab:red",
    "Branch 2000 PgC": "tab:orange",
    "Branch 750 PgC": "tab:purple",
    "esm-flat10-zec": "tab:green",

}

for label, fname in files.items():

    df = pd.read_csv(fname)

    temp = df[df["variable"] == "Surface Air Temperature Change"]

    year_cols = [c for c in temp.columns if str(c).isdigit()]
    years = np.array(year_cols, dtype=int)

    # shape = (n_years, n_members)
    arr = temp[year_cols].to_numpy(dtype=float).T

    mean = np.mean(arr, axis=1)
    p05 = np.percentile(arr, 5, axis=1)
    p95 = np.percentile(arr, 95, axis=1)

    ax.fill_between(
        years,
        p05,
        p95,
        color=colors[label],
        alpha=0.2,
    )

    ax.plot(
        years,
        mean,
        color=colors[label],
        lw=2,
        label=label,
    )

ax.set_xlabel("Year")
ax.set_ylabel("Surface air temperature change right, Deriv model (°C)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

plt.xlim([1750, 2400])
plt.ylim([0, 4])

