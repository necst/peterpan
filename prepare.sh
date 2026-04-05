#!/bin/bash

set -e

folder_name="PeterPan_STEP"
xclbin_name="PeterPan.xclbin"

pack_app_dir="build/PeterPan_APP"
app_src_dir="3DIRG_application"
bitstream_src="bitstreams/$xclbin_name"

step_dir="build/${folder_name}"

make -C sw clean
make config TASK=STEP

# STEP package: create only sw and common
rm -rf "$step_dir"
mkdir -p "$step_dir/sw" "$step_dir/common"

# Copy sw WITHOUT xrt.ini
if [ -d "./sw" ]; then
    rsync -a \
        --exclude 'xrt.ini' \
        ./sw/ "$step_dir/sw/"
else
    echo "Errore: cartella ./sw non trovata"
    exit 1
fi

# Copy xclbin into STEP/sw
if [ -f "$bitstream_src" ]; then
    cp "$bitstream_src" "$step_dir/sw/$xclbin_name"
else
    echo "Errore: xclbin non trovato: $bitstream_src"
    exit 1
fi

# Copy common
if [ -d "./common" ]; then
    rsync -a ./common/ "$step_dir/common/"
else
    echo "Errore: cartella ./common non trovata"
    exit 1
fi

# Copy default.cfg at the same level as sw and common
if [ -f "./default.cfg" ]; then
    cp "./default.cfg" "$step_dir/"
else
    echo "Errore: file ./default.cfg non trovato"
    exit 1
fi

# Add run script inside sw
if [ -f "./scripts/run_registration_step.sh" ]; then
    cp "./scripts/run_registration_step.sh" "$step_dir/sw/"
    chmod +x "$step_dir/sw/run_registration_step.sh"
else
    echo "Errore: file ./scripts/run_registration_step.sh non trovato"
    exit 1
fi

# Add Figure 6 script and requirements inside STEP/sw
if [ -f "./paper_fig/figure6/figure6.py" ]; then
    cp "./paper_fig/figure6/figure6.py" "$step_dir/sw/"
else
    echo "Errore: file ./paper_fig/figure6/figure6.py non trovato"
    exit 1
fi

if [ -f "./paper_fig/figure6/requirements.txt" ]; then
    cp "./paper_fig/figure6/requirements.txt" "$step_dir/sw/"
else
    echo "Errore: file ./paper_fig/figure6/requirements.txt non trovato"
    exit 1
fi

# Copy timing CSVs into STEP/sw
step_csv_src_dir="./paper_fig/csv"
if [ -d "$step_csv_src_dir" ]; then
    shopt -s nullglob
    step_csv_files=("$step_csv_src_dir"/*.csv)
    shopt -u nullglob

    if [ ${#step_csv_files[@]} -gt 0 ]; then
        cp "${step_csv_files[@]}" "$step_dir/sw/"
    else
        echo "Errore: nessun file CSV trovato in $step_csv_src_dir"
        exit 1
    fi
else
    echo "Errore: cartella $step_csv_src_dir non trovata"
    exit 1
fi

# ============================================================
# APP package
# ============================================================

make pack_peterpan_app \
    PACK_APP_DIR="$pack_app_dir" \
    APP_SRC_DIR="$app_src_dir" \
    BITSTREAM_SRC="$bitstream_src"

# Also copy the common folder into PeterPan_APP
if [ -d "./common" ]; then
    cp -r "./common" "$pack_app_dir/"
else
    echo "Errore: cartella ./common non trovata"
    exit 1
fi

# Copy dataset_downloader.sh at the same level as common and 3DIRG_application
if [ -f "./scripts/dataset_downloader.sh" ]; then
    cp "./scripts/dataset_downloader.sh" "$pack_app_dir/"
    chmod +x "$pack_app_dir/dataset_downloader.sh"
else
    echo "Errore: file ./scripts/dataset_downloader.sh non trovato"
    exit 1
fi

# In APP, the bitstream must be directly inside 3DIRG_application
app_dir="${pack_app_dir}/3DIRG_application"
if [ -d "$app_dir" ]; then
    cp "$bitstream_src" "${app_dir}/${xclbin_name}"
else
    echo "Errore: cartella applicazione non trovata: $app_dir"
    exit 1
fi

# Figure 7
app_figure7_dir="${app_dir}/Figure_7"
mkdir -p "$app_figure7_dir"

if [ -f "./paper_fig/figure7/figure7.py" ]; then
    cp "./paper_fig/figure7/figure7.py" "$app_figure7_dir/"
else
    echo "Errore: file ./paper_fig/figure7/figure7.py non trovato"
    exit 1
fi

if [ -f "./paper_fig/figure7/requirements.txt" ]; then
    cp "./paper_fig/figure7/requirements.txt" "$app_figure7_dir/"
else
    echo "Errore: file ./paper_fig/figure7/requirements.txt non trovato"
    exit 1
fi

# Figure 8
app_figure8_dir="${app_dir}/Figure_8"
mkdir -p "$app_figure8_dir"

if [ -f "./paper_fig/figure8/figure8.py" ]; then
    cp "./paper_fig/figure8/figure8.py" "$app_figure8_dir/"
else
    echo "Errore: file ./paper_fig/figure8/figure8.py non trovato"
    exit 1
fi

if [ -f "./paper_fig/figure8/requirements.txt" ]; then
    cp "./paper_fig/figure8/requirements.txt" "$app_figure8_dir/"
else
    echo "Errore: file ./paper_fig/figure8/requirements.txt non trovato"
    exit 1
fi

# Copy CSVs into Figure_8
figure8_csv_src_dir="./paper_fig/csv"
if [ -d "$figure8_csv_src_dir" ]; then
    shopt -s nullglob
    figure8_csv_files=("$figure8_csv_src_dir"/*.csv)
    shopt -u nullglob

    if [ ${#figure8_csv_files[@]} -gt 0 ]; then
        cp "${figure8_csv_files[@]}" "$app_figure8_dir/"
    else
        echo "Errore: nessun file CSV trovato in $figure8_csv_src_dir"
        exit 1
    fi
else
    echo "Errore: cartella $figure8_csv_src_dir non trovata"
    exit 1
fi

echo "--------------------"
echo "Packaging completed"
echo "- STEP package: $step_dir"
echo "  * ${step_dir}/sw"
echo "  * ${step_dir}/common"
echo "  * ${step_dir}/default.cfg"
echo "- AE package: $pack_app_dir"
echo "  * ${pack_app_dir}/dataset_downloader.sh"
echo "--------------------"

zip -r "build/${folder_name}.zip" "$step_dir"
zip -r "build/PeterPan_APP.zip" "$pack_app_dir"

export PATH=/usr/bin:$PATH