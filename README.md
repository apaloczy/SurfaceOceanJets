# SurfaceOceanJets

[![DOI](https://zenodo.org/badge/639956040.svg)](https://zenodo.org/badge/latestdoi/639956040)

This repository contains code for a manuscript titled **"Prevalence of deformation-scale surface currents"**, by A. Palóczy and J. H. LaCasce, [published in Geophysical Research Letters](https://doi.org/10.1029/2023GL104547). This [Jupyter notebook](https://nbviewer.jupyter.org/github/apaloczy/SurfaceOceanJets/blob/main/index.ipynb) provides an overview of the contents.

The directory `plot_figs/` contains Jupyter notebooks used to produce the figures in the manuscript (Figures 1-4 and S1-S2). These notebooks depend on the data files in the `data/` directory. Scripts to generate derived data files are in the `proc/` directory.

## Abstract

Understanding the transport of large ocean currents, like the Gulf Stream, has been of interest since the early days of oceanography. There has been less attention on the widths of the currents, although there exist several theoretical predictions. We present a census of time-averaged jet profiles, using _in situ_ and satellite data. The jets are typically asymmetrical, being narrower on the side with weaker stratification. The half-widths $L_j$ are correlated with the local deformation radius $L_d$ associated with the first surface mode on either side. The dependence of $L_j$ on $L_d$ is predicted by simple shallow water geostrophic adjustment models, with or without outcropping layers. This implies that potential vorticity is well-mixed adjacent to the jets, due most likely to mesoscale eddies. The findings suggest that surface jet widths are determined locally, by eddy-mean flow interactions.

## Authors
* [André Palóczy](https://www.mn.uio.no/geo/english/people/aca/metos/andrpalo/index.html) (<a.p.filho@geo.uio.no>)
* [Joseph H. LaCasce](https://www.mn.uio.no/geo/english/people/aca/metos/josepl/) (<j.h.lacasce@geo.uio.no>)

## Acknowledgments

AP and JHL acknowledge support from The Rough Ocean Project, funded by the Research Council of Norway under the Klimaforsk-programme, project \#302743. This study has been conducted using E.U. Copernicus Marine Service Information; DOI [https://doi.org/10.48670/moi-00145](https://doi.org/10.48670/moi-00145). We thank the United States National Science Foundation (NSF)'s Office of Polar Programs Antarctic Division (ANT) for support of the Drake Passage time series through Grants OPP-9816226, ANT-0338103, ANT-0838750, PLR-1341431, PLR-1542902 and ANT-2001646 and the Chereskin Lab at Scripps Institution of Oceanography/UCSD ([http://adcp.ucsd.edu/lmgould/](http://adcp.ucsd.edu/lmgould/)), for maintaining the collection, processing and dissemination of the ARSV Laurence M. Gould (LMG) ADCP data. We also thank Eric Firing and Jules Hummon (University of Hawaii) for their support of underway ADCP data through the UHDAS/CODAS software and the Joint Archive of Shipboard ADCP Data (JASADCP). We are grateful to the scientists and technicians onboard the LMG and R/Vs Ronald H. Brown, Atlantis and Thomas G. Thompson for ADCP data collection. Acquisition and processing of the GO-SHIP ADCP measurements has been funded by NSF grants OCE-0223505, OCE-0752970, and OCE-1437015. The XBT data were made available by the Scripps High Resolution XBT program ([www-hrx.ucsd.edu](www-hrx.ucsd.edu)). We thank the two anonymous reviewers for their input, which substantially improved the manuscript. We also thank Jonathan Lilly for creating the reformatted version of the TPJAOS dataset. Early discussions with Sarah Gille and Tom Rossby were very helpful.

## Open Research

Code required to reproduce the results and figures is available at [https://github.com/apaloczy/SurfaceOceanJets](https://github.com/apaloczy/SurfaceOceanJets), archived under DOI (DOI and in-line reference to be included). The objectively-mapped Absolute Dynamic Topography product is distributed by the Copernicus Marine Environment Monitoring Service (CMEMS, 2023). The DTU15 Mean Dynamic Topography is distributed by the Techincal University of Denmark (DTU, 2015). The merged along-track sea surface height anomaly TPJAOS dataset is distributed by NASA/PODAAC (Beckley et al., 2022), and the reformatted version used in this study was created by Lilly (2022). The Park et al. (2019) climatological Antarctic Circumpolar Current front positions dataset was created by Park and Durand (2019). The SRTM15+ bathymetry dataset is distributed by IGPP (2023). The ARSV Laurence M. Gould (LMG) underway ADCP velocity dataset is distributed by JASADCP (2022). The LMG underway XBT temperature dataset is distributed by the Scripps High Resolution XBT program (HRX, 2022). The GO-SHIP underway ADCP velocity datasets are distributed by GO-SHIP (2023).
