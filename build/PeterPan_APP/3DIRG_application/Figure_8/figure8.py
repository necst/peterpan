import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR

# ============================================================
# GLOBAL STYLE
# ============================================================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Palatino", "DejaVu Serif", "Times New Roman"],
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.handletextpad": 0.01,
    "hatch.linewidth": 0.6,
    "axes.labelpad": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# CANONICAL LABELS
# ============================================================
PETERPAN_1 = "PeterPan\N{SUBSCRIPT ONE}\nVCK5000"
PETERPAN_2 = "PeterPan\N{SUBSCRIPT TWO}\nVCK5000"
PETERPAN_ONLYPYR = "PeterPan-onlyPyr\nVCK5000"
PETERPAN_2_NOOPT = "PeterPan\N{SUBSCRIPT TWO} (no opt)\nVCK5000"
TRILLI_PYR = "TRILLI_PYR\nVCK5000"

# ============================================================
# 1) PALETTE (colorblind + print safe)
# ============================================================
COLOR_MAP = {
    # ITK
    "ITK": "#0072B2",
    "ITK_u7_o3": "#0072B2",
    "ITK\nu7_o3": "#0072B2",
    "ITK\nU7155H": "#0072B2",
    "ITK_U7155H": "#0072B2",

    # Matlab
    "Matlab": "#E69F00",
    "Matlab\ni7-13620H": "#E69F00",

    # SimpleITK
    "SimpleITK": "#009E73",
    "SITK-I7": "#009E73",
    "SimpleITK\nIntel i7-4770": "#009E73",

    # Hephaestus
    "Hephaestus": "#D55E00",
    "Hephaestus\nCPU + U280 ": "#D55E00",

    # Vitis
    "VitisLib\nVCK5000": "#CC79A7",

    # Kornia / GPU
    "Kornia\nA5000": "#56B4E9",
    "Kornia\nRTX4050": "#F0E442",
    "Kornia\nA100": "#999999",
    "Athena\nV100": "#CC79A7",

    # Athena labels (used in Fig.7c)
    "Athena\nA5000": "#56B4E9",
    "Athena\nA100": "#999999",
    "Athena\nRTX4050": "#F0E442",

    # TRILLI
    "TRILLI\nVCK5000": "#117733",
    "TRILLI\nVCK5000_128": "#44AA99",
    "TRILLI\nVCK5000_128_SW_PROG": "#AA4499",
    "TRILLI_PYR\nVCK5000": "#DDCC77",

    # PeterPan
    PETERPAN_2_NOOPT: "#332288",
    PETERPAN_ONLYPYR: "#668000",
    PETERPAN_2: "#CC5544",
    PETERPAN_1: "#88CCEE",
}

# ============================================================
# 2) HATCH MAP
# ============================================================
HATCH_MAP = {
    "ITK": "//",
    "ITK\nu7_o3": "//",
    "ITK\nU7155H": "//",
    "ITK_u7_o3": "*",

    "Matlab": "//",
    "Matlab\ni7-13620H": "//",

    "SimpleITK": "xx",
    "SimpleITK\nIntel i7-4770": "xx",
    "SITK-I7": "xx",

    "Hephaestus": "..",
    "Hephaestus\nCPU + U280 ": "..",

    "VitisLib\nVCK5000": "oo",

    "Kornia\nA5000": "\\\\",
    "Kornia\nRTX4050": "\\\\",
    "Kornia\nA100": "\\\\",
    "Athena\nV100": "\\\\",

    "Athena\nA5000": "\\\\",
    "Athena\nA100": "\\\\",
    "Athena\nRTX4050": "\\\\",

    "TRILLI\nVCK5000": "**",
    "TRILLI\nVCK5000_128": "--",
    "TRILLI\nVCK5000_128_SW_PROG": "||",
    "TRILLI_PYR\nVCK5000": "--",

    PETERPAN_2: "",
    PETERPAN_1: "",
    PETERPAN_ONLYPYR: "",
    PETERPAN_2_NOOPT: "",
}

# ============================================================
# 3) ABSOLUTE ORDER (shared)
# ============================================================
ORDER = [
    "ITK", "ITK_u7_o3", "ITK\nu7_o3", "ITK\nU7155H", "ITK_U7155H",
    "Matlab", "Matlab\ni7-13620H",
    "SimpleITK", "SITK-I7", "SimpleITK\nIntel i7-4770",
    "VitisLib\nVCK5000",
    "Hephaestus", "Hephaestus\nCPU + U280 ",
    "Kornia\nA5000", "Kornia\nRTX4050", "Kornia\nA100", "Athena\nV100",
    "TRILLI\nVCK5000", TRILLI_PYR, "TRILLI\nVCK5000_128", "TRILLI\nVCK5000_128_SW_PROG",
    PETERPAN_2_NOOPT, PETERPAN_1, PETERPAN_ONLYPYR, PETERPAN_2
]


class HandlerRectWithText(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):

        rect = mpatches.Rectangle(
            (xdescent, ydescent + height * 0.45),
            width,
            height * 0.40,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
            hatch=orig_handle.get_hatch(),
            linewidth=1,
            transform=trans,
        )

        txt = plt.Text(
            x=xdescent + width / 2,
            y=ydescent + height * 0.32,
            text=orig_handle.my_text,
            ha="center",
            va="top",
            fontsize=fontsize,
            transform=trans,
        )

        return [rect, txt]


# ============================================================
# Helpers
# ============================================================
def csv_path(*parts) -> str:
    return str(CSV_DIR.joinpath(*parts))


def norm(x: str) -> str:
    """Normalize config labels to canonical names used in COLOR_MAP/HATCH_MAP."""
    x = str(x)
    xl = x.lower()

    if "itk" in xl and "u7" in xl:
        return "ITK\nu7_o3"
    if "itk" in xl:
        return "ITK\nU7155H"

    if "matlab" in xl:
        return "Matlab\ni7-13620H"

    if "simpleitk" in xl or "sitk" in xl:
        return "SITK-I7"

    if "hephaestus" in xl:
        return "Hephaestus\nCPU + U280 "

    if "vitis" in xl:
        return "VitisLib\nVCK5000"

    if "a5000" in xl:
        return "Kornia\nA5000"
    if "rtx4050" in xl:
        return "Kornia\nRTX4050"
    if "a100" in xl:
        return "Kornia\nA100"
    if "v100" in xl:
        return "Athena\nV100"

    if "onlypyr" in xl:
        return PETERPAN_ONLYPYR
    if "peterpan_1" in xl or ("peterpan" in xl and "1level" in xl):
        return PETERPAN_1
    if "no_opt" in xl or "no opt" in xl:
        return PETERPAN_2_NOOPT
    if "peterpan_2" in xl:
        return PETERPAN_2
    if "peterpan" in xl:
        return PETERPAN_2

    if "trilli_pyr" in xl or ("trilli" in xl and "pyr" in xl):
        return TRILLI_PYR
    if "128_sw" in xl or "sw_prog" in xl:
        return "TRILLI\nVCK5000_128_SW_PROG"
    if "128" in xl and "trilli" in xl:
        return "TRILLI\nVCK5000_128"
    if "trilli" in xl:
        return "TRILLI\nVCK5000"

    return x


def load_clean_csv(path, sw_label, time_col):
    """Load csv, drop duplicated columns, rename time_col->Time, keep [Sw, Time]."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={time_col: "Time"})
    df = df[["Time"]].copy()
    df["Sw"] = sw_label
    df = df.reset_index(drop=True)
    return df[["Sw", "Time"]]


def read_time_csv_flexible(path, preferred_cols=("exec_time", "Time", "time", "tot", "tx", "ExeTime", "withPCIE_time")):
    """
    Read a CSV and pick the first matching time column from preferred_cols.
    Returns df with a single column 'Time'.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()]

    col_map = {c.lower(): c for c in df.columns}
    for c in preferred_cols:
        if c in df.columns:
            return df.rename(columns={c: "Time"})[["Time"]].copy()
        if c.lower() in col_map:
            real = col_map[c.lower()]
            return df.rename(columns={real: "Time"})[["Time"]].copy()

    raise ValueError(
        f"Could not find a time column in {path}. "
        f"Columns found: {list(df.columns)}. "
        f"Tried: {preferred_cols}"
    )


# ============================================================
# 5) GENERIC PLOTTER (single, clean)
# ============================================================
def make_plot(df, xcol, ycol, xlabel, filename,
              order_override=None,
              legend_ncol_override=None,
              legend_bbox_override=None,
              legend_labelspacing_override=None,
              legend_columnspacing_override=None,
              ylabel="Time [s]"):

    order_used = order_override if order_override is not None else ORDER
    present = [lab for lab in order_used if lab in df[xcol].unique()]

    df = df.copy()
    df[xcol] = pd.Categorical(df[xcol], categories=present, ordered=True)
    df = df.sort_values(xcol)

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = [COLOR_MAP[l] for l in present]
    sns.barplot(x=xcol, y=ycol, data=df, palette=palette, edgecolor="black", ax=ax)

    for i, bar in enumerate(ax.patches):
        if i < len(present):
            bar.set_hatch(HATCH_MAP.get(present[i], ""))

    if PETERPAN_2 in present:
        ref_label = PETERPAN_2
        ref = df[df[xcol] == ref_label][ycol].mean()
    else:
        min_idx = df[ycol].idxmin()
        ref_label = df.loc[min_idx, xcol]
        ref = df.loc[min_idx, ycol]

    for i, label in enumerate(present):
        val = df[df[xcol] == label][ycol].mean()
        if label == PETERPAN_2:
            txt = f"{val:.3f}s"
        else:
            txt = f"{val / ref:.2f}×"
        ax.text(i, val, txt, ha="center", va="bottom", fontsize=13)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_yscale("log")
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = []
    for l in present:
        r = mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=COLOR_MAP[l],
            edgecolor="black",
            hatch=HATCH_MAP.get(l, "")
        )
        r.my_text = l
        handles.append(r)

    ncol_legend = int(np.ceil(len(present) / 2)) if len(present) > 0 else 1
    if legend_ncol_override is not None:
        ncol_legend = legend_ncol_override

    bbox = (0.63, 0.98)
    if legend_bbox_override is not None:
        bbox = legend_bbox_override

    labelspacing_val = legend_labelspacing_override if legend_labelspacing_override is not None else 2
    columnspacing_val = legend_columnspacing_override if legend_columnspacing_override is not None else 3.5

    fig.legend(
        handles=handles,
        handler_map={mpatches.Rectangle: HandlerRectWithText()},
        loc="upper center",
        bbox_to_anchor=bbox,
        ncol=ncol_legend,
        frameon=False,
        fontsize=14,
        handleheight=2.4,
        handlelength=3.6,
        columnspacing=columnspacing_val,
        labelspacing=labelspacing_val,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(SCRIPT_DIR / f"{filename}.pdf"), bbox_inches="tight")
    fig.savefig(str(SCRIPT_DIR / f"{filename}.svg"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 7a — Transformation & Interpolation
# ============================================================
df_vitis = pd.read_csv(csv_path("", "VitisLibrary_246_ExecTime.csv"))
df_vitis = df_vitis.rename(columns={"Time": "Time"})
df_vitis["Config"] = "VitisLib\nVCK5000"

df_itk = pd.read_csv(csv_path("", "itk_u7_o3.csv"))
df_itk = df_itk.rename(columns={"TxTime": "Time"})
df_itk["Config"] = df_itk["Config"].apply(norm)

df_matlab = pd.read_csv(csv_path("", "matlab_i7_11.csv"))
df_matlab = df_matlab.rename(columns={"tx": "Time"})
df_matlab["Config"] = "Matlab\ni7-13620H"

df_heph = pd.read_csv(csv_path("", "hephaestus.csv"))
df_heph = df_heph.rename(columns={"TxTime": "Time"})
df_heph["Config"] = "Hephaestus\nCPU + U280 "

df_A5000 = pd.read_csv(csv_path("", "A5000_output_bilinear_512.csv"))
df_A5000 = df_A5000.rename(columns={"tx": "Time"})
df_A5000["Config"] = "Kornia\nA5000"

df_A100 = pd.read_csv(csv_path("", "A100_output_bilinear_512.csv"))
df_A100 = df_A100.rename(columns={"tx": "Time"})
df_A100["Config"] = "Kornia\nA100"

df_RTX = pd.read_csv(csv_path("", "RTX4050_output_bilinear_512.csv"))
df_RTX = df_RTX.rename(columns={"tx": "Time"})
df_RTX["Config"] = "Kornia\nRTX4050"

df_VCK_128 = pd.read_csv(csv_path("", "time_t08_aieplfreq_D512_N256_B032_I128_S16_TX_v202301_qdma.csv"))
df_VCK_128 = df_VCK_128.rename(columns={"exec_time": "Time"})
df_VCK_128["Config"] = "TRILLI\nVCK5000"

df_VCK_128_sw = pd.read_csv(csv_path("", "time_IPE128_D512_R512_C256_only_tx_sw_prog.csv"))
df_VCK_128_sw = df_VCK_128_sw.rename(columns={"exec_time": "Time"})
df_VCK_128_sw["Config"] = "TRILLI\nVCK5000_128_SW_PROG"

df_plot1 = pd.concat([
    df_itk[["Config", "Time"]],
    df_matlab[["Config", "Time"]],
    df_heph[["Config", "Time"]],
    df_vitis[["Config", "Time"]],
    df_A5000[["Config", "Time"]],
    df_A100[["Config", "Time"]],
    df_RTX[["Config", "Time"]],
    df_VCK_128[["Config", "Time"]],
    df_VCK_128_sw[["Config", "Time"]],
], ignore_index=True)

df_plot1.loc[df_plot1["Config"] == "TRILLI\nVCK5000_128_SW_PROG", "Config"] = PETERPAN_2

# ============================================================
# FIGURE 7b — Registration Step (TX + MI)
# ============================================================
df_itk2 = pd.read_csv(csv_path("", "itk_u7_o3.csv"))
df_itk2 = df_itk2.rename(columns={"ExeTime": "Time"})
df_itk2["Config"] = "ITK\nU7155H"

df_mat2 = pd.read_csv(csv_path("", "matlab_i7_11.csv"))
df_mat2 = df_mat2.rename(columns={"tot": "Time"})
df_mat2["Config"] = "Matlab\ni7-13620H"

df_heph2 = pd.read_csv(csv_path("", "hephaestus.csv"))
df_heph2["Config"] = "Hephaestus\nCPU + U280 "
df_heph2["Time"] = df_heph2["ExeTime"]

df_A5000_2 = pd.read_csv(csv_path("", "A5000_output_bilinear_512.csv"))
df_A5000_2 = df_A5000_2.rename(columns={"tot": "Time"})
df_A5000_2["Config"] = "Kornia\nA5000"

df_A100_2 = pd.read_csv(csv_path("", "A100_output_bilinear_512.csv"))
df_A100_2 = df_A100_2.rename(columns={"tot": "Time"})
df_A100_2["Config"] = "Kornia\nA100"

df_RTX_2 = pd.read_csv(csv_path("", "RTX4050_output_bilinear_512.csv"))
df_RTX_2 = df_RTX_2.rename(columns={"tot": "Time"})
df_RTX_2["Config"] = "Kornia\nRTX4050"

df_VCK2 = pd.read_csv(csv_path("", "TRILLI_VCK5000_128_IPE.csv"))
df_VCK2 = df_VCK2.rename(columns={"exec_time": "Time"})
df_VCK2["Config"] = "TRILLI\nVCK5000"

df_sw_noopt = pd.read_csv(csv_path("", "time_IPE128_D512_R512_C256_step.csv"))
df_sw_noopt = df_sw_noopt.rename(columns={"exec_time": "Time"})
df_sw_noopt["Config"] = PETERPAN_2_NOOPT

df_sw_opt = pd.read_csv(csv_path("", "time_IPE128_D512_R512_C256_step_opt.csv"))
df_sw_opt = df_sw_opt.rename(columns={"exec_time": "Time"})
df_sw_opt["Config"] = PETERPAN_2

PETERPAN_1_CSV = csv_path("", "time_IPE128_D512_R512_C256_peterpan1level.csv")
df_peter1_time = read_time_csv_flexible(PETERPAN_1_CSV, preferred_cols=("exec_time", "Time", "time", "tot"))
df_peter1_time["Config"] = PETERPAN_1
df_peter1 = df_peter1_time[["Config", "Time"]]

df_plot2 = pd.concat([
    df_itk2[["Config", "Time"]],
    df_mat2[["Config", "Time"]],
    df_heph2[["Config", "Time"]],
    df_A5000_2[["Config", "Time"]],
    df_A100_2[["Config", "Time"]],
    df_RTX_2[["Config", "Time"]],
    df_VCK2[["Config", "Time"]],
    df_peter1[["Config", "Time"]],
    df_sw_opt[["Config", "Time"]],
], ignore_index=True)

# ============================================================
# FIGURE 7c — Complete 3D Registration
# ============================================================
df_itkP = load_clean_csv(csv_path("", "itk_pow_estimate_o3.csv"), "ITK\nU7155H", "Time")
df_sitkP = load_clean_csv(csv_path("", "TimeSitkpowi7.csv"), "SITK-I7", "Time")
df_hephP = load_clean_csv(csv_path("", "TimePow_u280_8pe_16pen_2C_cache_noprint.csv"), "Hephaestus", "Time")

df_A5000P = load_clean_csv(csv_path("", "Powell_classicMoments_ampere.csv"), "Athena_A5000", "Time")
df_V100P = load_clean_csv(csv_path("", "Powell_classicMoments_mem_constr_oci.csv"), "Athena_V100", "Time")
df_A100P = load_clean_csv(csv_path("", "TimePowell_A100.csv"), "Athena_A100", "Time")
df_RTX4050P = load_clean_csv(csv_path("", "TimePowell_RTX4050.csv"), "Athena_RTX4050", "Time")

df_ICARUS = load_clean_csv(csv_path("", "TRILLI_3DIRG_pow.csv"), "TRILLI\nVCK5000", "withPCIE_time")

df_pyr2 = load_clean_csv(csv_path("", "only_pyr_timings_dz256_levels2.csv"), PETERPAN_ONLYPYR, "e2e_total_time_nosetup")
df_peter = load_clean_csv(csv_path("", "timings_dz32_levels4.csv"), PETERPAN_2, "e2e_total_time_nosetup")
df_trilli_pyr = load_clean_csv(
    csv_path("", "timings_dz32_levels4_pyr_no_sw_prog.csv"),
    TRILLI_PYR,
    "e2e_total_time_nosetup"
)

df_plot3 = pd.concat([
    df_itkP,
    df_sitkP,
    df_hephP,
    df_A5000P,
    df_V100P,
    df_A100P,
    df_RTX4050P,
    df_ICARUS,
    df_trilli_pyr,
    df_pyr2,
    df_peter
], ignore_index=True)

df_plot3["Config"] = df_plot3["Sw"].apply(norm)
df_plot3 = df_plot3[["Config", "Time"]]

df_plot3["Config"] = df_plot3["Config"].replace({
    "Kornia\nA5000": "Athena\nA5000",
    "Kornia\nA100": "Athena\nA100",
    "Kornia\nRTX4050": "Athena\nRTX4050",
})

ORDER_7C = []
for label in ORDER:
    if label == "ITK\nu7_o3":
        continue
    elif label == "Kornia\nA5000":
        ORDER_7C.append("Athena\nA5000")
    elif label == "Kornia\nA100":
        ORDER_7C.append("Athena\nA100")
    elif label == "Kornia\nRTX4050":
        ORDER_7C.append("Athena\nRTX4050")
    else:
        ORDER_7C.append(label)

# ============================================================
# GENERATE FIGURE 8 ONLY
# ============================================================
print("Generating Figure 8...")
make_plot(df_plot3, "Config", "Time",
          "Complete 3D Registration",
          "figure8",
          order_override=ORDER_7C,
          legend_ncol_override=5,
          legend_labelspacing_override=1.3,
          legend_columnspacing_override=3,
          legend_bbox_override=(0.65, 0.90),
          ylabel="End-To-End Execution Time [s]")

print("DONE — figure8.pdf / figure8.svg SAVED!")