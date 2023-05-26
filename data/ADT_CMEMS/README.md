ADT_CMEMS
---------

The regional subsets of the daily CMEMS L4 Absolute Dynamic Topography (ADT) maps (product **SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057**, [https://doi.org/10.48670/moi-00145](https://doi.org/10.48670/moi-00145), variables `adt`, `ugos` and `vgos`) should be placed in this directory.

Subsets of the daily ADT product can be downloaded from CMEMS using different methods. Below are the latitude and longitude bounds required for the files and the file names assumed by the processing scripts in this repository's `proc/` directory. All subset files should span the time period from **1993-01-01 00:00:00** through **2022-01-01 00:00:00**, except `CMEMS-ADT-DrakePassage_ACC.nc`, which only needs to cover the period from **1999-09-14 00:00:00** through **2018-12-20 00:00:00**

| File name                                      |      Longitude       |      Latitude        |
| :---                                           |        :---:         |       :---:          |
| CMEMS-ADT-stable_orbit-GulfStream.nc           | [281, 293]           | [28, 40]             |
| CMEMS-ADT-stable_orbit-KuroshioCurrent25N.nc   | [120.1875, 125.1875] | [22.4375, 27.4375]   |
| CMEMS-ADT-stable_orbit-KuroshioCurrent28p5N.nc | [124.6875, 129.6875] | [25.9375, 30.9375]   |
| CMEMS-ADT-stable_orbit-BrazilCurrent29S.nc     | [310.1875, 315.1875] | [-31.5625, -26.5625] |
| CMEMS-ADT-stable_orbit-AgulhasCurrent.nc       | [24, 30]             | [-36, -32]           |
| CMEMS-ADT-stable_orbit-EAC29S.nc               | [151.4375, 156.4375] | [-31.5625, -26.5625] |
| CMEMS-ADT-DrakePassage_ACC.nc                  | [292, 319]           | [-65.0, -54.5]       |
