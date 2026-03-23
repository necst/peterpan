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
# PALETTE
# ============================================================
COLOR_MAP = {
    "ITK": "#0072B2",
    "ITK_u7_o3": "#0072B2",
    "ITK\nu7_o3": "#0072B2",
    "ITK\nU7155H": "#0072B2",
    "ITK_U7155H": "#0072B2",

    "Matlab": "#E69F00",
    "Matlab\ni7-13620H": "#E69F00",

    "SimpleITK": "#009E73",
    "SITK-I7": "#009E73",
    "SimpleITK\nIntel i7-4770": "#009E73",

    "Hephaestus": "#D55E00",
    "Hephaestus\nCPU + U280 ": "#D55E00",

    "VitisLib\nVCK5000": "#CC79A7",

    "Kornia\nA5000": "#56B4E9",
    "Kornia\nRTX4050": "#F0E442",
    "Kornia\nA100": "#999999",
    "Athena\nV100": "#CC79A7",

    "Athena\nA5000": "#56B4E9",
    "Athena\nA100": "#999999",
    "Athena\nRTX4050": "#F0E442",

    "TRILLI\nVCK5000": "#117733",
    "TRILLI\nVCK5000_128": "#44AA99",
    "TRILLI\nVCK5000_128_SW_PROG": "#AA4499",
    "TRILLI_PYR\nVCK5000": "#DDCC77",

    PETERPAN_2_NOOPT: "#332288",
    PETERPAN_ONLYPYR: "#668000",
    PETERPAN_2: "#CC5544",
    PETERPAN_1: "#88CCEE",
}

# ============================================================
# HATCH MAP
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
# ORDER
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


def csv_path(*parts) -> str:
    return str(CSV_DIR.joinpath(*parts))


def make_plot(df, xcol, ycol, xlabel, filename, ylabel="Time [s]"):
    present = [lab for lab in ORDER if lab in df[xcol].unique()]

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

    fig.legend(
        handles=handles,
        handler_map={mpatches.Rectangle: HandlerRectWithText()},
        loc="upper center",
        bbox_to_anchor=(0.63, 0.98),
        ncol=ncol_legend,
        frameon=False,
        fontsize=14,
        handleheight=2.4,
        handlelength=3.6,
        columnspacing=3.5,
        labelspacing=2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(SCRIPT_DIR / f"{filename}.pdf"), bbox_inches="tight")
    fig.savefig(str(SCRIPT_DIR / f"{filename}.svg"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 6 = old FIGURE 7a — Transformation & Interpolation
# ============================================================
df_vitis = pd.read_csv(csv_path("time", "VitisLibrary_246_ExecTime.csv"))
df_vitis["Config"] = "VitisLib\nVCK5000"

df_itk = pd.read_csv(csv_path("time", "itk_u7_o3.csv"))
df_itk = df_itk.rename(columns={"TxTime": "Time"})
df_itk["Config"] = "ITK\nu7_o3"

df_matlab = pd.read_csv(csv_path("time", "matlab_i7_11.csv"))
df_matlab = df_matlab.rename(columns={"tx": "Time"})
df_matlab["Config"] = "Matlab\ni7-13620H"

df_heph = pd.read_csv(csv_path("time", "hephaestus.csv"))
df_heph = df_heph.rename(columns={"TxTime": "Time"})
df_heph["Config"] = "Hephaestus\nCPU + U280 "

df_A5000 = pd.read_csv(csv_path("time", "A5000_output_bilinear_512.csv"))
df_A5000 = df_A5000.rename(columns={"tx": "Time"})
df_A5000["Config"] = "Kornia\nA5000"

df_A100 = pd.read_csv(csv_path("time", "A100_output_bilinear_512.csv"))
df_A100 = df_A100.rename(columns={"tx": "Time"})
df_A100["Config"] = "Kornia\nA100"

df_RTX = pd.read_csv(csv_path("time", "RTX4050_output_bilinear_512.csv"))
df_RTX = df_RTX.rename(columns={"tx": "Time"})
df_RTX["Config"] = "Kornia\nRTX4050"

df_VCK_128 = pd.read_csv(csv_path("time", "time_t08_aieplfreq_D512_N256_B032_I128_S16_TX_v202301_qdma.csv"))
df_VCK_128 = df_VCK_128.rename(columns={"exec_time": "Time"})
df_VCK_128["Config"] = "TRILLI\nVCK5000"

df_VCK_128_sw = pd.read_csv(csv_path("time", "time_IPE128_D512_R512_C256_only_tx_sw_prog.csv"))
df_VCK_128_sw = df_VCK_128_sw.rename(columns={"exec_time": "Time"})
df_VCK_128_sw["Config"] = PETERPAN_2

df_plot = pd.concat([
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

print("Generating Figure 6...")
make_plot(
    df_plot,
    "Config",
    "Time",
    "Transformation & Interpolation",
    "figure6",
    ylabel="Time [s]"
)
print("DONE — figure6.pdf / figure6.svg saved!")