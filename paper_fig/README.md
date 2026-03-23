# 📊 Paper Figures

This folder contains several subfolder, one per each result figure in the paper. 
Note that provided codes require pdfTex installed to properly render the final chart. Also, we 
suggest to use the specific tool versions for ensure the correct cart rendering

```
pip install -r requirements.txt 
```

While, for pdfTeX we suggest running:

```
sudo apt update
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

You can check correctness of this command, by using the command: 
```
latex --version
```

The output should looks like
```
pdfTeX 3.141592653-2.6-1.40.22 (TeX Live 2022/dev/Debian)
kpathsea version 6.3.4/dev
Copyright 2021 Han The Thanh (pdfTeX) et al.
There is NO warranty.  Redistribution of this software is
covered by the terms of both the pdfTeX copyright and
the Lesser GNU General Public License.
For more information about these matters, see the file
named COPYING and the pdfTeX source.
Primary author of pdfTeX: Han The Thanh (pdfTeX) et al.
Compiled with libpng 1.6.37; using libpng 1.6.37
Compiled with zlib 1.2.11; using zlib 1.2.11
Compiled with xpdf version 4.03
```

## 📌 Overview
Following, we give an overview of each subfolder. More details are provided in each subfolder README.md. 
Per each subfolder, we provide enough data to reproduce the results. However, custom ones can be produced following the testing workflow in the main README (root folder).

- *figure6:* This folder contains code and data for Figure 6 of the paper, thatanalyzes TRILLI's different configuration by scaling number of Interpolation Processing Elements (IPEs) and volume Depths. 

- *figure7:* This folder contains code and data for Figure 7 of the paper, that compares TRILLI against literature solution considering both Transformation, Registration Step (Transformation + Mutual Information), and complete image registration application.

- *figure8:* This folder contains code and data for Figure 8 of the paper, that shows TRILLI's worst-case error upon registration, by using a single slice from the registered volume to depict the difference between ideal and registered images.
