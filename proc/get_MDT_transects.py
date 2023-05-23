# Read DTU15 MDT and locate jet transects.
import numpy as np
import matplotlib.pyplot as plt
from xarray import open_dataset
from scipy.interpolate import interpn
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from ap_tools.utils import get_xtrackline, lon180to360, lon360to180, near2, rot_vec
from gsw import distance, grav
from gsw import f as fcor
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere


def get_contour(lon, lat, A, isoA, contour_idx=None, cyclic=False, smooth=False, window_length=21, win_type='hann', **kw):
    """
    USAGE
    -----
    lon_isob, lat_isob = get_isobath(lon, lat, A, iso, cyclic=False, smooth=False, window_length=21, win_type='hann', **kw)

    Retrieves the 'lon_isob','lat_isob' coordinates of a wanted 'isoA' isoline from a topography array 'topo', with 'lon_topo','lat_topo' coordinates.
    """
    lon, lat, A = map(np.array, (lon, lat, A))

    fig, ax = plt.subplots()
    cs = ax.contour(lon, lat, A, [isoA])
    coll = cs.collections[0]
    ## Test all lines to find the longest one.
    ## This is assumed to be the wanted contour.
    ncoll = len(coll.get_paths())
    siz = np.array([])
    for n in range(ncoll):
        path = coll.get_paths()[n]
        siz = np.append(siz, path.vertices.shape[0])

    if contour_idx is None: # Take the longest contour.
        f = np.argmax(siz)
    else: # Get a specific contour instead.
        f = contour_idx

    xiso = coll.get_paths()[f].vertices[:, 0]
    yiso = coll.get_paths()[f].vertices[:, 1]
    plt.close()

    # Smooth the isobath with a moving window.
    # Periodize according to window length to avoid losing edges.
    if smooth:
        fleft = window_length//2
        fright = -window_length//2 + 1
        if cyclic:
            xl = xiso[:fleft] + 360
            xr = xiso[fright:] - 360
            yl = yiso[:fleft]
            yr = yiso[fright:]
            xiso = np.concatenate((xr, xiso, xl))
            yiso = np.concatenate((yr, yiso, yl))

    return xiso, yiso


def compass2trig(ang):
    ang = np.array(ang, ndmin=1)
    ang -= 90                       # Move origin to east.
    ang = (ang + 360)%360           # Wrap negative angles back to 360.
    ang = 360 - ang                 # Make positive couter-clockwise.
    ang = ((ang + 180) % 360) - 180 # [0 360] to [-180 180].
    ang[ang==360] = 0

    return ang


def mdt2uv(lon, lat, mdt):
    deglat2m = 1852*60
    dy = np.gradient(lat, axis=0)*deglat2m
    dx = np.gradient(lon, axis=1)*np.cos(lat*np.pi/180)*deglat2m
    dmdtdy, dmdtdx = np.gradient(mdt)
    dmdtdy, dmdtdx = dmdtdy/dy, dmdtdx/dx
    ugfac = grav(lat, 0)/fcor(lat)
    u = -ugfac*dmdtdy
    v = +ugfac*dmdtdx

    return u, v


#---
plt.close("all")

GET_NEAREST_TRACK_TO_CMEMS_MDT = True

wanted_lon, wanted_lat, sup, d = -69, 38, "Gulf Stream", 10
# wanted_lon, wanted_lat, sup, d = -76.65, 33, "Gulf Stream 33N", 5
# wanted_lon, wanted_lat, sup, d = 29, -36, "Agulhas Current", 6
# wanted_lon, wanted_lat, sup, d, GET_NEAREST_TRACK_TO_CMEMS_MDT = -48, -30.12, "Brazil Current 29S", 5, False
# wanted_lon, wanted_lat, sup, d = 154, -29, "East Australian Current 29S", 5
# wanted_lon, wanted_lat, sup, d = 122.7, 25, "Kuroshio Current 25N", 5
# wanted_lon, wanted_lat, sup, d = 127.23, 28.5, "Kuroshio Current 28p5N", 5

SAVE_NPZ = True
L = 200

if sup in ["Gulf Stream 33N", "Brazil Current 29S", "Kuroshio Current 28p5N", "Kuroshio Current 25N"]:
    L = 150

if sup in ["East Australian Current 29S"]:
    L = 130

if sup=="Agulhas Current":
    contour_idx = 1
else:
    contour_idx = None

dL = 5
Uthresh = 0.7 # [m/s]

proj = ccrs.PlateCarree()
figsize = (8, 8)
f = "../data/DTU15/DTU15MDT_1min.mdt.nc"

# Plot global MDT and geostrophic velocity maps.
xmi, xma = wanted_lon - d, wanted_lon + d
ymi, yma = wanted_lat - d, wanted_lat + d

ds = open_dataset(f).sel(dict(lon=slice(lon180to360(xmi)[0], lon180to360(xma)[0]), lat=slice(ymi, yma)))
mdtbb = ds["mdt"].values
lonbb, latbb = lon360to180(ds["lon"].values), ds["lat"].values
lonbb, latbb = np.meshgrid(lonbb, latbb)
lonbb0, latbb0 = lonbb[0, :], latbb[:, 0]

ubb, vbb = mdt2uv(lonbb, latbb, mdtbb)
Ubb = np.sqrt(ubb**2 + vbb**2)
fbad = Ubb>Uthresh
Ubb[fbad] = np.nan
ubb[fbad] = np.nan
vbb[fbad] = np.nan

# Get bathymetry.
fbathymetry = "../data/srtm15p/SRTM15_V2.5.5.nc"
dstopo = open_dataset(fbathymetry).sel(lon=slice(xmi, xma), lat=slice(ymi, yma))
ztopo = - dstopo.interp(coords=dict(lon=lonbb0, lat=latbb0))["z"].values

min_depth = 0 # [m]
fbadtopo = ztopo<min_depth
Ubb[fbadtopo] = np.nan
ubb[fbadtopo] = np.nan
vbb[fbadtopo] = np.nan

vmi, vma = np.nanmin(mdtbb), np.nanmax(mdtbb)
Uma = np.nanmax(Ubb)

figdir = "MDTtransects/"

# Cross-stream profile of geostrophic speed.
# 1) Find point of maximum geostrophic speed within domain (or an exact point, if specified).
# 2) Draw MDT contour closest to max speed, get angle of that contour.
# 3) Draw line across the mean jet orientation.
fnameCMEMS = sup.replace(" ", "-") + "_xtream_transect.npz"
try:
    dCMEMSmdt = np.load("../data/derived/" + fnameCMEMS)
except FileNotFoundError:
    print("No CMEMS MDT file.")
    dCMEMSmdt = None

# Use old CMEMS MDT lines and find closes DTU15 MDT lines instead of prescribed (lon, lat).
if GET_NEAREST_TRACK_TO_CMEMS_MDT and dCMEMSmdt is not None:
    wanted_lon, wanted_lat = dCMEMSmdt["lonma"].flatten()[0], dCMEMSmdt["latma"].flatten()[0]

if dCMEMSmdt is not None:
    xperp_CMEMS, yperp_CMEMS = dCMEMSmdt["xperp"], dCMEMSmdt["yperp"]

lonma, latma = near2(lonbb, latbb, wanted_lon, wanted_lat)
fbb2 = np.logical_and(lonbb==lonma, latbb==latma)
mdt0 = mdtbb[fbb2][0]

xmdtma, ymdtma = get_contour(lonbb, latbb, mdtbb, mdt0, contour_idx=contour_idx)
xmdtma, ymdtma = np.flipud(xmdtma), np.flipud(ymdtma)
xmdtmam, ymdtmam = 0.5*(xmdtma[1:] + xmdtma[:-1]), 0.5*(ymdtma[1:] + ymdtma[:-1])

Uperp_on_mdt_contour = interpn((lonbb0, latbb0), Ubb, (xmdtma, ymdtma), method="linear")
f0 = near2(xmdtma, ymdtma, wanted_lon, wanted_lat, return_index=True)[0]

x1, x2, y1, y2 = xmdtma[f0-1], xmdtma[f0+1], ymdtma[f0-1], ymdtma[f0+1]
xperp, yperp = get_xtrackline(x1, x2, y1, y2, L=L, dL=dL)

fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=figsize)
cs = ax.pcolormesh(lonbb, latbb, Ubb, vmin=0, vmax=Uma, zorder=7)
ax.contour(lonbb, latbb, mdtbb, levels=20, colors="gray", linestyles="solid", linewidths=0.5, zorder=8)
ax.plot(xmdtma, ymdtma, "r", linewidth=0.5, zorder=9)
ax.plot(xperp, yperp, "m", linewidth=1.5, zorder=9)
if dCMEMSmdt is not None:
    ax.plot(xperp_CMEMS, yperp_CMEMS, "m:", linewidth=1.5, zorder=9)
ax.coastlines()
ax.set_xticks([wanted_lon], crs=proj)
ax.set_yticks([wanted_lat], crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
cb = fig.colorbar(cs, ax=ax, orientation="vertical")
cb.set_label(r"$U_g$ [m/s]", fontsize=16)
ax.set_title(sup + " cross-stream transect", fontsize=12, fontweight="black")
fig.savefig(figdir + sup.replace(" ", "-") + "_MDT-transect.png", bbox_inches="tight")

# Interpolate Ug along the cross-stream line.
ui = interpn((lonbb0, latbb0), ubb, (xperp, yperp), method="linear")
vi = interpn((lonbb0, latbb0), vbb, (xperp, yperp), method="linear")
distperp = np.append(0, np.cumsum(distance(xperp, yperp)))*1e-3
fpeak = np.nanargmax(np.sqrt(ui**2 + vi**2))
distperp = distperp - distperp[fpeak]

# Rotate to transect orientation.
p1, p2 = LatLon_sphere(yperp[0], xperp[0]), LatLon_sphere(yperp[-1], xperp[-1])
angrot = compass2trig(p1.initialBearingTo(p2))[0]
uirot, virot = rot_vec(ui, vi, angle=angrot)
if -np.nanmin(virot) > np.nanmax(virot):
    angrot += 180
    uirot, virot = rot_vec(ui, vi, angle=angrot)


# Save (lat, lon) position, Lj/Ld and cross-stream transect for use with ADT movies and Lj-Ld scatterplot.
if SAVE_NPZ:
    dsave = dict(lonma=lonma, latma=latma, xperp=xperp, yperp=yperp, distperp=distperp, Uperp=virot, name=sup)
    fnameout = sup.replace(" ", "-") + "_xstream_transect_DTU15.npz"
    np.savez(fnameout, **dsave)


plt.show(block=False)
