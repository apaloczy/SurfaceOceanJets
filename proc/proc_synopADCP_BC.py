import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from hdf5storage import loadmat
from xarray import open_dataset, DataArray, Dataset
from pygeodesy.sphericalNvector import LatLon
from ap_tools.utils import rot_vec, compass2trig, lon360to180, bbox2ij
from gsw import distance
from scipy.interpolate import griddata, interp1d
import cartopy.crs as ccrs

#---
plt.close("all")
head = "../data/synopADCP/"

INTERPOLATE_LOW_SHIPSPD = False
PLOT = True
FLIP_ANG = True

# Read data, extract transect and rotate to cross-transect orientation.
cruise = "2011_A10"
ts, te = "2011-10-27 09:00:00", "2011-10-29 21:50:00"
dist_split = (388, 535)#(366, 496) # [km]
zbot_avg = 50 # [m]

ds = open_dataset(glob(head + cruise + "/" + cruise + "*os75bb_short.nc")[0])
dstide = open_dataset(glob(head + cruise + "/" + cruise + "*os75bb_tide_tpxo72.nc")[0])
ds = ds.sel(dict(time=slice(ts, te)))
dstide = dstide.sel(dict(time=slice(ts, te)))

time, lon, lat, z = ds["time"], ds["lon"], ds["lat"], ds["depth"]
lon, lat = lon.interpolate_na("time").values, lat.interpolate_na("time").values
hdng = ds["heading"]
shipspd = np.sqrt(ds["uship"]**2 + ds["vship"]**2)
u, v = ds["u"], ds["v"]
ut, vt = dstide["u"], dstide["v"]

# Subtract barotropic tidal velocities.
u0, v0 = u.copy(), v.copy()
u = u - ut
v = v - vt

# ## Plot tidal velocities.
# fig, ax = plt.subplots()
# ut.plot(ax=ax)
# vt.plot(ax=ax)

# 1) Plot figures to make sure the right transect was extracted.
lonaux, lataux = lon[np.isfinite(lon)], lat[np.isfinite(lat)]
if PLOT:
    u0["depth_cell"] = -u0["depth_cell"]
    v0["depth_cell"] = -v0["depth_cell"]
    u["depth_cell"] = -u["depth_cell"]
    v["depth_cell"] = -v["depth_cell"]
    us, vs = u.isel(dict(depth_cell=0)).values, v.isel(dict(depth_cell=0)).values
    sub = 10

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax1, ax2, ax3 = ax[0]
    ax4, ax5, ax6 = ax[1]
    u.T.plot(ax=ax1)
    (u0 - u).T.plot(ax=ax2) # Barotropic tidal velocities.
    hdng.plot(ax=ax3)
    ax3r = ax3.twinx()
    shipspd.plot(color="r", ax=ax3r)
    v.T.plot(ax=ax4)
    (v0 - v).T.plot(ax=ax5) # Barotropic tidal velocities.
    ax6.plot(lon, lat, "k", zorder=8)
    ax6.quiver(lon[::sub], lat[::sub], us[::sub], vs[::sub], zorder=9)
    ax6.plot(lonaux[0], lataux[0], marker="o", ms=15, mfc="g", mec="g", zorder=-9)
    ax6.plot(lonaux[-1], lataux[-1], marker="x", ms=15, mfc="r", mec="r", mew=5, zorder=-9)
    fig.suptitle("Original velocities")

# 2) Rotate velocities based on transect angle.
ps, pe = LatLon(lataux[0], lonaux[0]), LatLon(lataux[-1], lonaux[-1])
if FLIP_ANG:
    ps, pe = pe, ps

hdngse = ps.initialBearingTo(pe)
uaux, vaux = rot_vec(u, v, angle=compass2trig(hdngse)) #Single value, as opposed to hdng.values[:, np.newaxis].
ualong, vcross = u.copy(), v.copy()
ualong.values, vcross.values = uaux, vaux

if PLOT:
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax1, ax2 = ax[0]
    ax3, ax4 = ax[1]
    u.T.plot(ax=ax1)
    v.T.plot(ax=ax3)
    ualong.T.plot(ax=ax2)
    vcross.T.plot(ax=ax4)
    ax1.set_title("Zonal/meridional velocities")
    ax2.set_title("Along-/cross-transect velocities")

# 3) Depth-average near the surface and mask/interpolate data when ship was on stations.
fnzavg = z.values > zbot_avg
ualong_zavg, vcross_zavg = ualong.values.copy(), vcross.values.copy()
ualong_zavg[fnzavg], vcross_zavg[fnzavg] = np.nan, np.nan
ualong_zavg, vcross_zavg = np.nanmean(ualong_zavg, axis=1), np.nanmean(vcross_zavg, axis=1)

shipspd_thresh = 2 # [m/s]
fbad_shipspd = shipspd.values < shipspd_thresh
ualong_zavg_aux, vcross_zavg_aux = ualong_zavg.copy(), vcross_zavg.copy()
ualong_zavg_aux[fbad_shipspd] = np.nan
vcross_zavg_aux[fbad_shipspd] = np.nan

dist = np.append(0, np.cumsum(distance(lon, lat)))*1e-3

if INTERPOLATE_LOW_SHIPSPD:
    ualong_zavg_aux, vcross_zavg_aux = DataArray(ualong_zavg_aux, coords=dict(dist=dist)), DataArray(vcross_zavg_aux, coords=dict(dist=dist))
    ualong_zavg2 = ualong_zavg_aux.interpolate_na(dim="dist").values
    vcross_zavg2 = vcross_zavg_aux.interpolate_na(dim="dist").values
else:
    ualong_zavg2 = ualong_zavg_aux.copy()
    vcross_zavg2 = vcross_zavg_aux.copy()

if PLOT:
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    ax1, ax2 = ax
    ax1r = ax1.twinx()
    ax1.plot(dist, ualong_zavg, "b", label="$u$")
    ax1.plot(dist, vcross_zavg, "r", label="$v$")
    ax1r.plot(dist, shipspd.values, "gray")
    ax1.set_xlim(dist[0], dist[-1])
    ax1.axhline(color="k", linestyle="dashed")
    ax1.legend()

    ax2r = ax2.twinx()
    ax2.plot(dist, ualong_zavg, "b", linewidth=2, alpha=0.2)
    ax2.plot(dist, vcross_zavg, "r", linewidth=2, alpha=0.2)

    ax2.plot(dist, ualong_zavg2, "b", linewidth=0.4)
    ax2.plot(dist, vcross_zavg2, "r", linewidth=0.4)

    ax2r.plot(dist, shipspd.values, "gray", alpha=0.2)
    ax2.set_xlim(dist[0], dist[-1])
    ax2.axhline(color="k", linestyle="dashed")

    ax1.set_ylabel("Velocity [m/s]", y=0)
    ax1r.set_ylabel("Ship speed [m/s]", y=0)

    ax1.set_title("Depth-averaged velocities (top %d m)"%zbot_avg)
    ax2.set_xlabel("Distance [km]")

# 4) Extract BC jet.
fBC = np.logical_and(dist>=dist_split[0], dist<=dist_split[1])
distBC = dist[fBC]
lonBC = lon[fBC]
latBC = lat[fBC]
fshpspd = fbad_shipspd[fBC]

vBC = vcross_zavg2[fBC]
vBC = - vBC

# 5) Project the data on a straight transect and remap the velocity.
numll = lonBC.size
lonBCp = np.linspace(lonBC[0], lonBC[-1], num=numll)
latBCp = np.linspace(latBC[0], latBC[-1], num=numll)
distBCp = np.append(0, np.cumsum(distance(lonBCp, latBCp)))*1e-3

distBCo, vBCo = distBC.copy(), vBC.copy()
distBCo = distBCo - distBCo[0]
vBC = interp1d(lonBC[~fshpspd], vBCo[~fshpspd], bounds_error=False)(lonBCp)
xBCo = distBCo - distBCo[np.nanargmax(vBCo)]
xBC = distBCp - distBCp[np.nanargmax(vBC)]

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.plot(lonBC, latBC, "b")
ax.plot(lonBCp, latBCp, "r")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

fig, ax = plt.subplots()
ax.plot(xBCo, vBCo)
ax.plot(xBC, vBC)
xl, xr = xBC[0], xBC[-1]
ax.set_xlim(xl, xr)
ax.axhline(color="gray", linestyle="solid")
ax.axvline(color="gray", linestyle="solid")
ax.set_title("Depth-averaged velocities (top %d m)"%zbot_avg)


# 5) Get Lds from LaCasce & Groeskamp's (2020) map.
dd = loadmat("../data/misc/Ld_LaCasce-Groeskamp_2020.mat")
lonLd, latLd, Ldflati, Ldsurfi = dd["xt"].squeeze(), dd["yt"].squeeze(), dd["Ld_flat_RK4"].squeeze().T*1e-3, dd["Ld_rough_RK4"].squeeze().T*1e-3
xiLd, yiLd = np.meshgrid(lon360to180(lonLd), latLd)

dl = 5
xmin, xmax = np.nanmin(lon) - dl, np.nanmax(lon) + dl
ymin, ymax = np.nanmin(lat) - dl, np.nanmax(lat) + dl
jmin, jmax, imin, imax = bbox2ij(xiLd, yiLd, bbox=[xmin, xmax, ymin, ymax])
xiLd, yiLd = xiLd[imin:imax, jmin:jmax], yiLd[imin:imax, jmin:jmax]
Ldflati, Ldsurfi = Ldflati[imin:imax, jmin:jmax], Ldsurfi[imin:imax, jmin:jmax]

fgf = np.isfinite(Ldflati)
fgs = np.isfinite(Ldsurfi)
Ldflat_BC = griddata((xiLd[fgf], yiLd[fgf]), Ldflati[fgf], (lonBCp, latBCp), method="cubic")
Ldsurf_BC = griddata((xiLd[fgs], yiLd[fgs]), Ldsurfi[fgs], (lonBCp, latBCp), method="cubic")

# Quick plot map of Ld.
vminflat = np.nanmin(Ldflati)#_BC)
vmaxflat = np.nanmax(Ldflati)#_BC)
vminsurf = np.nanmin(Ldsurfi)#_BC)
vmaxsurf = np.nanmax(Ldsurfi)#_BC)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
cs1 = ax1.pcolormesh(xiLd, yiLd, Ldflati, vmin=vminflat, vmax=vmaxflat)
cs2 = ax2.pcolormesh(xiLd, yiLd, Ldsurfi, vmin=vminsurf, vmax=vmaxsurf)
ax1.plot(lonBC, latBC, "r")
ax2.plot(lonBC, latBC, "r")
fig.colorbar(cs1, ax=ax1)
fig.colorbar(cs2, ax=ax2)

# Plot values of Ld along ship track.

fig, ax = plt.subplots()
ax.plot(xBC, Ldflat_BC, "b", label="Flat")
ax.plot(xBC, Ldsurf_BC, "r", label="Surface")
ax.grid(); ax.legend()
ax.set_xlabel("Distance [km]")
ax.set_ylabel("$L_d$ [km]")

# 6) Save velocity profile and deformation radius values to do exponential fits.
coords = dict(x=xBC)
Lon = DataArray(lonBC, coords=coords, attrs=dict(units="Degrees East", long_name="Longitude"))
Lat = DataArray(latBC, coords=coords, attrs=dict(units="Degrees North", long_name="Latitude"))
U = DataArray(vBC, coords=coords, attrs=dict(units="m/s", long_name="Downstream velocity"))
Ldflat_BC = DataArray(Ldflat_BC, coords=coords, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf_BC = DataArray(Ldsurf_BC, coords=coords, attrs=dict(units="km", long_name="First surface deformation radius"))
dsout = Dataset(data_vars=dict(u=U, lon=Lon, lat=Lat, Ldflat=Ldflat_BC, Ldsurf=Ldsurf_BC))

# Flip so shallow side is on the left.
dsout["x"] = - dsout["x"]
dsout = dsout.sortby("x")

dsout.to_netcdf("../data/derived/BC_synop.nc")

plt.show(block=False)
