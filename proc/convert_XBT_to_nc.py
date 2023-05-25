# Reads in XBT data in .10 ASCII format and
# saves it as an xarray Dataset.
import numpy as np
from xarray import IndexVariable, DataArray, Dataset
from pandas import Timestamp
from gsw import distance
from glob import glob
from ap_tools.utils import lon360to180


def read_header(f):
    l = f.readline().strip("\n")
    lat, lon = float(l[:11]), lon360to180(float(l[11:19]))[0]
    dd, mo, yy = map(int, (l[19:21], l[22:24], l[25:27]))
    hh, mm, ss = map(int, (l[28:30], l[31:33], l[34:36]))
    if yy<90:
        yy += 2000
    else:
        yy += 1900
    ti = Timestamp("%d-%d-%d %d:%d:%d"%(yy, mo, dd, hh, mm, ss))

    return ti, lon, lat, f


def read_profile(f):
    T = []
    for n in range(8):
        l = f.readline().strip("\n").split(" ")
        for c in l:
            if c!="":
                if c=="-9999":
                    T.append(np.nan)
                else:
                    T.append(int(c)/1000)

    return np.array(T), f


#---
head = "../data/XBT_LMG/"

z = - np.arange(5, 905, 10) # Depth axis.
z = IndexVariable("z", z, attrs=dict(units="m"))

fnames = glob(head + "*.10")
fnames.sort()
nf = len(fnames)

nn = 1
for fname in fnames:
    print("Reading %s, file %d / %d"%(fname.split("/")[-1], nn, nf))
    f = open(fname, mode="r")
    l = f.readline()
    nprofs = int(l.strip(" ").strip("\n"))

    # Read header and all profiles in each cruise.
    for n in range(nprofs):
        ti, loni, lati, f = read_header(f)
        Ti, f = read_profile(f)
        if n==0:
            tn, lonn, latn = ti, loni, lati
            Tn = Ti[:, np.newaxis]
        else:
            Tn = np.hstack((Tn, Ti[:, np.newaxis]))
            tn = np.append(tn, ti)
            lonn = np.append(lonn, loni)
            latn = np.append(latn, lati)

    # Put all XBT profiles and coordinates in a Dataset and save in netCDF.
    dn = np.append(0, np.cumsum(distance(lonn, latn)))*1e-3 # [km].
    tn = IndexVariable("x", tn, attrs=dict(timezone="GMT"))
    lonn = IndexVariable("x", lonn, attrs=dict(units="Degrees east"))
    latn = IndexVariable("x", latn, attrs=dict(units="Degrees north"))
    dn = IndexVariable("x", dn, attrs=dict(units="km", long_name="Along-transect distance"))
    Tn = DataArray(data=Tn, coords=(z, dn), dims=("z", "x"), attrs=dict(units="Degrees Celsius"))
    ds = Dataset(dict(T=Tn), coords=dict(z=z, x=dn, lon=lonn, lat=latn, t=tn))
    ds.to_netcdf(fname.replace(".10", ".nc")) # Save in netCDF.

    f.close()
    nn += 1
