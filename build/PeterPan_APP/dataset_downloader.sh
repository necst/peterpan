#!/bin/bash
# ============================================================
#  Dataset Downloader Script
#  Requirements: curl, unzip
# ============================================================

set -euo pipefail

OUTPUT_DIR="."
mkdir -p "$OUTPUT_DIR"

download_gdrive_file() {
    local file_id="$1"
    local output="$2"

    echo "  Downloading: $output"

    # Download with confirmation bypass for large files
    curl -sL \
        -o "$output" \
        -b /dev/null \
        -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

    local size
    size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
    echo "    -> $(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes")"
}

echo "============================================"
echo "  Dataset Downloader "
echo "============================================"
echo ""

# File IDs
FILE1_ID="1Gock4gHE86x-mHUR4jxq0mM6LtwUeCRy"
FILE2_ID="11kdPxlW7-r2KirlUf23vLirGx5NdugAa"

# Step 1: Download
echo "[1/2] Downloading files..."
download_gdrive_file "$FILE1_ID" "$OUTPUT_DIR/file1.zip"
download_gdrive_file "$FILE2_ID" "$OUTPUT_DIR/file2.zip"

# Step 2: Extract
echo ""
echo "[2/2] Extracting..."
for zip in "$OUTPUT_DIR"/*.zip; do
    echo "  Unzipping: $(basename "$zip")"
    unzip -o -q "$zip" -d "$OUTPUT_DIR"
    rm "$zip"
done


mv CT FIXED
mv PET MOVING
echo ""
echo "============================================"
echo "  Done! Files saved to: $OUTPUT_DIR"
echo "============================================"
echo ""
