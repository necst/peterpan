import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcdefaults()
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
plt.rcParams["font.size"] = 12

# === READ CSV FILES FROM CURRENT FOLDER ===
csv_files = sorted(glob.glob("./output/timings*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the current folder.")

all_dfs = []

for file in csv_files:
    match = re.search(r"[dD][zZ]?(\d+)", file)
    if not match:
        raise ValueError(f"Unable to extract Dz from filename: {file}")
    dz_value = int(match.group(1))

    df = pd.read_csv(file)
    df["SubvolumeDepth"] = dz_value
    all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# === COMPUTE AVERAGES FOR LEVEL 1 (USING Lvl1_total) ===
level1 = data[data["PyramidalLevels"] == 1]
mean_level1 = level1.groupby("SubvolumeDepth")["Lvl1_total"].mean()

# === FILTER LEVELS > 1 ===
data = data[data["PyramidalLevels"] > 1]

# === COMPUTE MEAN AND STD ===
agg = (
    data.groupby(["PyramidalLevels", "SubvolumeDepth"])["e2e_total_time_nosetup"]
    .agg(["mean", "std"])
    .reset_index()
)

y_min = np.min(agg["mean"] - agg["std"])
y_max = np.max(agg["mean"] + agg["std"])

plt.figure(figsize=(10, 4))

for dz in sorted(agg["SubvolumeDepth"].unique()):
    subset = agg[agg["SubvolumeDepth"] == dz]

    plt.errorbar(
        subset["PyramidalLevels"],
        subset["mean"],
        yerr=subset["std"],
        fmt="o",
        capsize=6,
        markersize=14,
        elinewidth=2,
        markeredgewidth=1.5,
        markeredgecolor="black",
        label=f"$D_{{z}}={dz}$"
    )

plt.xlabel(r"\# Pyramidal levels", fontsize=18)
plt.ylabel("Execution Time [s]", fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

red_handles = []
red_labels = []

line_styles = {
    0: "--",
    1: ":"
}

colors = ["blue", "red"]
for i, dz in enumerate(sorted(mean_level1.index)):
    y = mean_level1.loc[dz]
    linestyle = line_styles.get(i, "--")
    color = colors[i] if i < len(colors) else "black"

    plt.axhline(y=y, color=color, linestyle=linestyle, linewidth=2)

    proxy = Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2)
    red_handles.append(proxy)
    red_labels.append(f"level1_$D_{{z}}={dz}$")

handles, labels = plt.gca().get_legend_handles_labels()
handles.extend(red_handles)
labels.extend(red_labels)

plt.legend(handles, labels, fontsize=18, ncol=2)


plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("e2e_nosetup_points.pdf")

# ============================================================
# ======= COMPUTE MAXIMUM SPEEDUP (USING Lvl1_total) ====
# ============================================================

df_lvl1 = mean_level1.reset_index()
df_lvl1.columns = ["SubvolumeDepth", "Time_level1"]

best = agg.loc[agg["mean"].idxmin()]

best_time = best["mean"]
best_lvl = best["PyramidalLevels"]
best_dz = best["SubvolumeDepth"]

t_level1 = df_lvl1[df_lvl1["SubvolumeDepth"] == best_dz]["Time_level1"].values[0]

speedup = t_level1 / best_time

print("\n===== SPEEDUP RESULT =====")
print(f"Maximum speedup: {speedup:.2f}x")
print(f"Fastest level: PyramidalLevels = {best_lvl}")
print(f"SubvolumeDepth (Dz): {best_dz}")
print(f"Level-1 time (Lvl1_total): {t_level1:.3f} s")
print(f"Minimum time found: {best_time:.3f} s")