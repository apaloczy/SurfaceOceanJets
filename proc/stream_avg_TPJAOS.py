# Calculate stream-averaged geostrophic velocity from
# merged along-track gridded altimetry (JASON-TOPEX-OSTM-JASON3).
import numpy as np
import matplotlib.pyplot as plt
from gsw import distance
from xarray import open_dataset, DataArray, Dataset, IndexVariable
from pandas import Timestamp
from hdf5storage import loadmat
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator, griddata
from scipy.stats import circmean
from statistics import mode
from ap_tools.utils import lon360to180, lon180to360, near, near2, rot_vec, compass2trig
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import LAND
from gsw import f as fcor
from gsw import grav
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere
from cmocean.cm import balance
from os import system
from os.path import isdir
proj = ccrs.PlateCarree()


def rolltrim(x, n):
    if n!=0:
        x = np.roll(x, n)
        if n>0:
            x[:n] = np.nan
        else:
            x[n:] = np.nan
    else:
        pass

    return x


def polydiff(p):
    n = len(p) - 1 # Degree of parent polynomial.

    return p[:-1]*np.arange(n, 0, -1)


def pltmaps_Ld(xperp, yperp, sup, name, nlr, Ldflatl, Ldflatr, Ldsurfl, Ldsurfr, dxy=3, SAVEFIG=True):
    global xiLd, yiLd, Ldflati, Ldsurfi, xt, yt, zt, isobs

    fm = xperp.size//2
    xperp = lon180to360(xperp)
    xm, ym = xperp[fm], yperp[fm]
    xl, yl = xperp[fm - nlr], yperp[fm - nlr]
    xr, yr = xperp[fm + nlr], yperp[fm + nlr]
    xperp = lon360to180(xperp)
    ext = (xm - dxy, xm + dxy, ym - dxy, ym + dxy)
    fsub = np.logical_and(np.logical_and(xiLd>=ext[0], xiLd<=ext[1]), np.logical_and(yiLd>=ext[2], yiLd<=ext[3]))
    Ldflatisub, Ldsurfisub = Ldflati[fsub], Ldsurfi[fsub]
    vmi, vma = np.minimum(np.nanmin(Ldflatisub), np.nanmin(Ldsurfisub)), np.maximum(np.nanmax(Ldflatisub), np.nanmax(Ldsurfisub))
    vmi = np.maximum(vmi, 1)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw=dict(projection=proj))
    ax1, ax2 = ax

    ax1.set_extent(ext, crs=proj)
    cs1 = ax1.pcolormesh(xiLd, yiLd, Ldflati, zorder=8, cmap=plt.cm.jet, vmin=vmi, vmax=vma)
    ax1.add_feature(LAND, zorder=9, color="k")
    cc = ax1.contour(xt, yt, zt, levels=isobs, colors="gray", zorder=9)
    ax1.clabel(cc, fontsize=5)
    ax1.plot(xperp, yperp, "m", zorder=10)
    ax1.plot(lon360to180(xm), ym, marker="o", ms=3, mfc="m", mec="m", mew=0.5, zorder=10)
    ax1.plot(lon360to180(xl), yl, marker="o", ms=3, mfc="r", mec="w", mew=0.5, zorder=10)
    ax1.plot(lon360to180(xr), yr, marker="o", ms=3, mfc="g", mec="w", mew=0.5, zorder=10)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax1.set_title(" $L_{dl}, L_{dr}$ = %.1f km, %.1f km"%(Ldflatl, Ldflatr), fontsize=10, fontweight="black")
    cax = ax1.inset_axes((0, -0.15, 1, 0.05))
    cb = fig.colorbar(cs1, cax=cax, orientation="horizontal")
    cb.set_label("Flat $L_d$ [km]", fontsize=12, fontweight="black")

    ax2.set_extent(ext, crs=proj)
    cs2 = ax2.pcolormesh(xiLd, yiLd, Ldsurfi, zorder=8, cmap=plt.cm.jet, vmin=vmi, vmax=vma)
    ax2.add_feature(LAND, zorder=9, color="k")
    cc = ax2.contour(xt, yt, zt, levels=isobs, colors="gray", zorder=9)
    ax2.clabel(cc, fontsize=5)
    ax2.plot(xperp, yperp, "m", zorder=10)
    ax2.plot(lon360to180(xm), ym, marker="o", ms=3, mfc="m", mec="m", mew=0.5, zorder=10)
    ax2.plot(lon360to180(xl), yl, marker="o", ms=3, mfc="r", mec="w", mew=0.5, zorder=10)
    ax2.plot(lon360to180(xr), yr, marker="o", ms=3, mfc="g", mec="w", mew=0.5, zorder=10)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax2.set_title(" $L_{dl}, L_{dr}$ = %.1f km, %.1f km"%(Ldsurfl, Ldsurfr), fontsize=10, fontweight="black")
    cax = ax2.inset_axes((0, -0.15, 1, 0.05))
    cb = fig.colorbar(cs2, cax=cax, orientation="horizontal")
    cb.set_label("Surface $L_d$ [km]", fontsize=12, fontweight="black")
    fig.suptitle(sup, fontsize=13, fontweight="black", y=0.875)
    # fig.tight_layout()
    if SAVEFIG:
        fig.savefig("%s/Ld_map_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=200)


def get_xtrackline_from_angle(lon0, lat0, ang, L=200, dL=10):
    km2m = 1e3
    L = L/2
    L, dL = L*km2m, dL*km2m
    nh = int(L/dL)
    pm = LatLon_sphere(lat0, lon0)
    ang = (90 - ang) + 90
    angb = ang + 180

    # Create perpendicular line starting from the midpoint.
    N = range(1, nh + 1)
    p = []
    _ = [p.append(pm.destination(dL*n, angb)) for n in N]
    p.reverse()
    p.append(pm)
    _ = [p.append(pm.destination(dL*n, ang)) for n in N]

    lon = np.array([p.lon for p in p])
    lat = np.array([p.lat for p in p])

    return lon, lat


def get_trkang(lon1, lon2, lat1, lat2):
    pt1 = LatLon_sphere(lat1, lon1)
    pt2 = LatLon_sphere(lat2, lon2)
    if (lat2 - lat1)<0:
        pt1, pt2 = pt2, pt1
    ang = compass2trig(pt1.bearingTo(pt2))[0] - 90 # Angle *y-axis* (direction normal to satellite track) makes with true east (positive CCW).

    return ang


#---
plt.close("all")

PLT_SPAGHETTI_PROFILES = True # Whether to plot all along-track ADT profiles in a single plot.
PLT_INDIVIDUAL_PROFILES = True # Whether to plot the ADT profiles from each cycle on individual plots.

Llr = 30 # [km]
EXCLUDE_FLAGGED_POINTS = True
min_dfc, max_dfc = 0, 1000
poly_deg, poly_deg_low = 9, 5 # Degree of polynomial fit for along-track ADT to find origin.
MASK_FIT_POLY_SLA, fdiff_poly_thresh = True, 0.1
GET_L4_ADT_STREAM_VELOCITY = True # Whether to calculate the stream-averaged velocity from the L4 CMEMS ADT for comparison.

USE_ADT_FOR_EXACT_REFERENCE_ORIGIN = False # Whether to use L4 CMEMS ADT to find instantaneous jet core instead of the max along-track slope from the along-track ADT.
USE_ADT_TO_SEARCH_REFERENCE_ORIGIN = True # Whether to use L4 CMEMS ADT to find instantaneous jet core from searching the max along-track slope from the along-track ADT within a search radius about the mat L4 ADT velocity.

FLIP_CURRENTS = ["AgulhasCurrent", "BrazilCurrent29S"]
ROTATE_USING_ADT = True
exec(open("ug_stream_params.py").read())

f = "tpjaos_rf.nc"
head = "../data/merged_alongtrack_altimetry/"

Lx = 300        # Total length of desired transect [km]
Lxuvang = 30    # Half-distance in km about the jet core to determine rotation angle from.
LxoriginL4 = 50 # Half-distance in km about the jet core in the L4 ADT to determine core position in the L2 ADT.
Ldmin = 1.0     # [km]

headmdt = "../data/derived/"

print("")
print("======================")
print(sup)
print("======================")
print("")

namedir = sup.replace(" ", "")
if isdir(namedir):
    system("rm %s/all_profiles_rotateADT_TPJAOS/*.png"%namedir)
    system("rm %s/all_profiles_TPJAOS/*.png"%namedir)
else:
    system("mkdir %s"%namedir)
    system("mkdir %s/all_profiles_rotateADT_TPJAOS"%namedir)
    system("mkdir %s/all_profiles_TPJAOS"%namedir)

dx_approx = 5.76420450685313 # [km]
nx = round(Lx/dx_approx)
nangpts_fit = round(Lxuvang/dx_approx)
nangpts_search_originL4 = round(LxoriginL4/dx_approx)
nxh = nx//2
nnx = nx + 1
wanted_lon180 = wanted_lon
wanted_lon = lon180to360(wanted_lon)[0]

wanted_lon180 = wanted_lon180 + xnudge
wanted_lon = wanted_lon + xnudge
wanted_lat = wanted_lat + ynudge

ds0 = open_dataset(head + f)

# Fix single NaT on the time axis.
taux = ds0["cycle_time"].values
fnat = np.where(~np.isfinite(taux))[0][0]
dtaux = (taux[fnat + 1] - taux[fnat - 1])/2
taux[fnat] = taux[fnat - 1] + dtaux

ds0["cycle_time"] = taux.copy()
# ds0 = ds0.sel(dict(cycle_time=slice("2015-01-01 00:00:00", "2015-03-01 00:00:00")))
lon0, lat0 = ds0["lon"], ds0["lat"]

# Get bathymetry.
fbathymetry = "../data/srtm15p/SRTM15_V2.5.5.nc"
isobs = (200, 2000, 4000)
d = 5
xmi, xma = wanted_lon180 - d, wanted_lon180 + d
ymi, yma = wanted_lat - d, wanted_lat + d
ds = open_dataset(fbathymetry).sel(lon=slice(xmi, xma), lat=slice(ymi, yma))
xt, yt = np.meshgrid(ds["lon"].values, ds["lat"].values)
zt = -ds["z"].values

if ROTATE_USING_ADT:
    if "GulfStream" in name:
        nameADT = "GulfStream"
    else:
        nameADT = name
    headADT = "../data/ADT_CMEMS/"
    fADT = "CMEMS-ADT-stable_orbit-" + nameADT + ".nc"

    dsADT = open_dataset(headADT + fADT)
    xADT, yADT = dsADT["longitude"].values, dsADT["latitude"].values
    xx, yy = np.meshgrid(xADT, yADT)
    xyADT = (xx.ravel(), yy.ravel())

    # Read cross-stream transect from CMEMS MDT to get the jet orientation.
    d = np.load(headmdt + fname_xstream)
    lonmdt, latmdt = d["xperp"], d["yperp"]
    nxhmdt = lonmdt.size//2
    angmdt = get_trkang(lonmdt[nxhmdt], lonmdt[nxhmdt+1], latmdt[nxhmdt], latmdt[nxhmdt+1])
    if angmdt<0:
        angmdt = -angmdt

    # Overlay ground tracks on ADT to find the track most perpendicular to the time-averaged jet's orientation.
    fig, ax = plt.subplots()
    dsADT["adt"].mean("time").plot(ax=ax)
    cc = ax.contour(xt, yt, zt, levels=isobs, colors="gray")
    ax.clabel(cc)
    trklon, trklat = lon360to180(lon0.values), lat0.values
    fbb = np.logical_and(np.logical_and(trklon>=xADT.min(), trklon<=xADT.max()), np.logical_and(trklat>=yADT.min(), trklat<=yADT.max()))

    ax.plot(trklon[fbb], trklat[fbb], linestyle="none", marker="o", ms=1, mfc="k", mec="k")
    ax.plot(lonmdt, latmdt, "gray")
    ax.plot(lonmdt[nxhmdt], latmdt[nxhmdt], marker="s", ms=6, mfc="gray", mec="w")
    ax.plot(wanted_lon180, wanted_lat, marker="s", ms=6, mfc="gray", mec="w")

    lon0aux, lat0aux = lon360to180(lon0.values), lat0.values
    trklonaux, trklataux = lon0aux.copy(), lat0aux.copy()
    # Find nearest point and see if angle is within wanted limits relative to mean jet orientation.
    dfc = ds0["dfc"].values
    angdiff = np.inf
    ftrk0, fx0 = near2(trklonaux, trklataux, wanted_lon180, wanted_lat, return_index=True, npts=1)
    while angdiff>=maxangmdt_thresh:
        ftrk, fx = near2(trklonaux, trklataux, wanted_lon180, wanted_lat, return_index=True, npts=1)
        trklonaux[ftrk, fx], trklataux[ftrk, fx] = np.nan, np.nan
        dfci = dfc[ftrk, fx]
        if dfci<min_dfc:
            print("Selected point too close to the coast (%d km < %d km). Skipping."%(dfci, min_dfc))
            angdiff = np.inf
        elif dfci>max_dfc:
            print("Selected point too far from the coast (%d km > %d km). Skipping."%(dfci, max_dfc))
            angdiff = np.inf
        else:
            angtrkaux = get_trkang(lon0aux[ftrk, fx], lon0aux[ftrk, fx+1], lat0aux[ftrk, fx], lat0aux[ftrk, fx+1])
            if angtrkaux<0:
                angtrkaux += 180
            angdiff = np.abs(angmdt - angtrkaux)
            print(angmdt, angtrkaux, angdiff)
    ax.plot(lon0aux[ftrk, fx], lat0aux[ftrk, fx], marker="s", ms=6, mfc="m", mec="w")
    ax.set_title("Time-averaged ADT")
    plt.show(block=False)
    fig.savefig("%s/track_orientation_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight")
else: # Find the orbit that is closest to wanted (lon, lat).
    ftrk, fx = near2(lon0.values, lat0.values, wanted_lon, wanted_lat, return_index=True, npts=1) # Find nearest point in (orbit number, along-track position) space.

# Take slice about origin.
ds = ds0.sel(dict(track_number=ftrk, along_track=slice(fx - nxh, fx + nxh + 1)))
lon360 = ds["lon"].values
lon, lat = lon360to180(lon360), ds["lat"].values
xyalongtrk = (lon, lat)
dtoff = ds["time_offset"].values[nxh]
ncycles = ds["cycle_time"].size

# Get the orientation of the satellite track.
angtrk = get_trkang(lon[nxh], lon[nxh+1], lat[nxh], lat[nxh+1])

x0 = np.append(0, np.cumsum(distance(lon, lat))) # [m]
dx = np.gradient(x0, edge_order=2) # [m]
dxa = 0.5*(dx[1:] + dx[:-1])
x0 = x0 - x0[nxh]
x0m = x0.copy()
x0 = x0*1e-3 # [km]
ds = ds.update(dict(x0=("along_track", x0))).set_coords("x0")
lat0 = ds["lat"].mean().values.flatten()[0]
f0 = fcor(lat0)
g0 = grav(lat0, 0)
ugfac = g0/f0
Rofac = g0/f0**2

dsla_thresh = np.abs(us_thresh/ugfac) # [m/m]
ddsla_thresh = Ro_thresh/Rofac        # [m/m2]

# Then add the DTU15 MDT to obtain the instantaneous ADT.
dd = 3
xl, xr = lon360.min() - dd, lon360.max() + dd
yl, yr = lat.min() - dd, lat.max() + dd
bb = dict(lon=slice(xl, xr), lat=slice(yl, yr))

fmdt = "../data/DTU15/DTU15MDT_1min.mdt.nc"
dsmdt = open_dataset(fmdt).sel(bb)

Imdt = RegularGridInterpolator((dsmdt["lat"].values, dsmdt["lon"].values), dsmdt["mdt"].values, method="cubic")
mdt = np.array([Imdt((yi, xi)) for yi, xi in zip(lat, lon360)])

# Add the MDT to the SLA relative to the MSS to get the (along-track) ADT.
ds["sla"].values = ds["sla"].values + mdt

if EXCLUDE_FLAGGED_POINTS:
    pbad_flags_all = []
    flags_all = None
    for n in range(ncycles):
        cyc = n + 1
        dsi = ds.isel(dict(cycle_time=n))
        fbflags = np.logical_and(dsi["flag"].values >= flagl, dsi["flag"].values <= flagr)
        pbad_flags = 100*np.sum(fbflags)/nnx # < 4096 should pass the three last flags: (Distance_to_Land<50km, Water_Depth<200m, Single_Frequency_Altimeter).
        print("%1.1f %% suspicious data points (Cycle %d) rejected"%(pbad_flags, cyc))
        pbad_flags_all.append(pbad_flags)
        if flags_all is not None:
            flags_all = np.vstack((flags_all, dsi["flag"].values[np.newaxis, :]))
        else:
            flags_all = dsi["flag"].values[np.newaxis, :]
    print("")
    print("Mean (std) flagged points across all cycles: %1.1f %% (%1.1f %%)"%(np.mean(pbad_flags_all), np.std(pbad_flags_all)))
    print("Most common flag: %1.1f"%mode(flags_all.ravel().tolist()))
    print("")

# Loop over all cycles to find the origin of the jet-following
# reference frameand interpolate.
if PLT_SPAGHETTI_PROFILES:
    fig0, ax0 = plt.subplots()
auxx, auxx2 = [], []
slas = None
for n in range(ncycles):
    cyc = n + 1
    print("Cycle %d of %d"%(cyc, ncycles))
    dsi = ds.isel(dict(cycle_time=n))
    fgflags = ~np.logical_and(dsi["flag"].values >= flagl, dsi["flag"].values <= flagr)
    if EXCLUDE_FLAGGED_POINTS:
        slai = dsi["sla"].where(fgflags).values # NaN out flagged points.
    else:
        slai = dsi["sla"].values
    ti = dsi["cycle_time"].values + dtoff

    pgud = 100*np.isfinite(slai).sum()/nnx
    if pgud < pgud_thresh:
        print("Less than %.1f%% (%.1f%%) good data points. Skipping this cycle."%(pgud_thresh, pgud))
        continue

    # Use polynomial fit just to find the max slope.
    fg = np.isfinite(slai)
    x0aux = x0m[fg]
    pslai_low = np.polyfit(x0aux, slai[fg], poly_deg_low)
    pslai = np.polyfit(x0aux, slai[fg], poly_deg)
    pdslai = polydiff(pslai)
    pddslai = polydiff(pdslai)
    slai_fit_low = np.polyval(pslai_low, x0m)
    slai_fit = np.polyval(pslai, x0m)
    dslai_fit_aux = np.polyval(pdslai, x0m)
    ddslai_fit_aux = np.polyval(pddslai, x0m)
    slai_fit_aux = slai_fit.copy()
    if MASK_FIT_POLY_SLA:
        fdiff_poly = np.abs(slai_fit - slai_fit_low) > fdiff_poly_thresh
        slai_fit_aux[fdiff_poly] = np.nan
        dslai_fit_aux[fdiff_poly] = np.nan
        ddslai_fit_aux[fdiff_poly] = np.nan
    ftrim = np.where(fg)[0]
    ftl, ftr = ftrim[0], ftrim[-1]
    if ftl>0:
        slai_fit_aux[:ftl] = np.nan
        dslai_fit_aux[:ftl+1] = np.nan
        ddslai_fit_aux[:ftl+2] = np.nan
    if ftr<nx:
        slai_fit_aux[ftr:] = np.nan
        dslai_fit_aux[ftr-1:] = np.nan
        ddslai_fit_aux[ftr-2:] = np.nan

    fspk1 = np.abs(slai_fit_aux)>ADT_thresh
    fspk2 = np.abs(dslai_fit_aux)>dsla_thresh
    fspk3 = np.abs(ddslai_fit_aux)>ddsla_thresh

    if fspk1.any():
        slai_fit_aux[fspk1] = np.nan
        dslai_fit_aux[fspk1] = np.nan
        ddslai_fit_aux[fspk1] = np.nan
        print("Rejected %d points above ADT threshold."%fspk1.sum())
    if fspk2.any():
        slai_fit_aux[fspk2] = np.nan
        dslai_fit_aux[fspk2] = np.nan
        ddslai_fit_aux[fspk2] = np.nan
        print("Rejected %d points above ug threshold."%fspk2.sum())
    if fspk3.any():
        slai_fit_aux[fspk3] = np.nan
        dslai_fit_aux[fspk3] = np.nan
        ddslai_fit_aux[fspk3] = np.nan
        print("Rejected %d points above d(ug)/dx threshold."%fspk3.sum())

    auxx.append(dslai_fit_aux)
    auxx2.append(np.gradient(slai_fit_aux, edge_order=2)/dx)

    if name=="EAC29S":
        fc_trk = np.nanargmax(np.abs(dslai_fit_aux))
    else:
        fc_trk = np.nanargmin(dslai_fit_aux)

    while fc_trk==0 or fc_trk==nx:
        slai_fit_aux[fc_trk] = np.nan
        dslai_fit_aux[fc_trk] = np.nan
        ddslai_fit_aux[fc_trk] = np.nan

        # fc_trk = np.nanargmax(np.abs(dslai_fit_aux))
        if name=="EAC29S":
            fc_trk = np.nanargmax(np.abs(dslai_fit_aux))
        else:
            fc_trk = np.nanargmin(dslai_fit_aux)

    # Get an instantaneous rotation angle using the
    # surface absolute geostrophic velocity derived
    # from the CMEMS ADT.
    if ROTATE_USING_ADT:
        dsiADT = dsADT.interp(time=ti, method="linear")
        if np.isnan(dsiADT["adt"].values).all():
            print("All-NaN ADT. Skipping coordinate rotation for this pass.")
            continue

        # Interpolate u, v to along-track transect and determine angle.
        uiADT, viADT = dsiADT["ugos"].values, dsiADT["vgos"].values
        uiADTa, viADTa = uiADT.ravel(), viADT.ravel()
        ui = griddata(xyADT, uiADTa, xyalongtrk, method="linear")
        vi = griddata(xyADT, viADTa, xyalongtrk, method="linear")
        if np.isnan(ui).all() or np.isnan(vi).all():
            print("All-NaN interpolated (u, v). Skipping coordinate rotation for this pass.")
            continue

        # Find origin of stream reference frame from ADT.
        angradt = angtrk - 90
        uir, vir = rot_vec(ui, vi, angle=angradt)
        if name in FLIP_CURRENTS:
            vir_aux = -vir
        else:
            vir_aux = vir
        fc_adt = np.nanargmax(vir_aux)

        if USE_ADT_FOR_EXACT_REFERENCE_ORIGIN:
            fc = fc_adt # Use exact position of max vel from L4 ADT.
        elif USE_ADT_TO_SEARCH_REFERENCE_ORIGIN:
            fladt = np.maximum(0, fc_adt - nangpts_search_originL4)
            fradt = np.minimum(nx, fc_adt + nangpts_search_originL4 + 1)
            try:
                if name=="EAC29S":
                    fc = np.where(dslai_fit_aux==np.nanmax(np.abs(dslai_fit_aux[fladt:fradt])))[0][0]
                else:
                    fc = np.where(dslai_fit_aux==np.nanmin(dslai_fit_aux[fladt:fradt]))[0][0]
            except:
                print("All-NaN slice, using origin from L2 ADT away from L4 ADT*****")
                fc = fc_trk
        else:
            fc = fc_trk

        fl = np.maximum(0, fc - nangpts_fit)
        fr = np.minimum(nx, fc + nangpts_fit + 1)
        uic = np.nanmean(ui[fl:fr])
        vic = np.nanmean(vi[fl:fr])
        if np.isnan(uic) or np.isnan(vic):
            print("All-NaN along-track (u, v). Skipping coordinate rotation for this pass.")
            continue

        angc = np.arctan2(vic, uic)*180/np.pi # Angle relative to true east.
        angr = angc - angtrk # Rotation angle that transforms cross-track velocity to instantaneous jet frame of reference.
        xcperp, ycperp = get_xtrackline_from_angle(lon[fc], lat[fc], angc, L=Lx, dL=dx_approx)

        # Plot ADT with surface geostrophic velocity vectors and MDT transect.
        fig, ax = plt.subplots()
        dsiADT["adt"].plot(ax=ax, cmap=balance)
        cc = ax.contour(xt, yt, zt, levels=isobs, colors="gray", linewidths=1)
        ax.clabel(cc)
        ax.quiver(xADT, yADT, uiADT, viADT)
        ax.plot(lon, lat, "m") # TPJAOS ground track.
        ax.plot(xcperp, ycperp, "k")
        ax.plot(lon[fl:fr], lat[fl:fr], "w", linestyle="dashed")

        # Interpolate CMEMS ADT velocity to its own streamwise frame
        # For comparison with the along-track ADT velocities.
        if GET_L4_ADT_STREAM_VELOCITY:
            if not USE_ADT_FOR_EXACT_REFERENCE_ORIGIN: # Recalculate based on L4 ADT origin if not already done.
                fl = np.maximum(0, fc_adt - nangpts_fit)
                fr = np.minimum(nx, fc_adt + nangpts_fit + 1)
                uic = np.nanmean(ui[fl:fr])
                vic = np.nanmean(vi[fl:fr])
                if np.isnan(uic) or np.isnan(vic):
                    print("All-NaN along-track (u, v). Skipping coordinate rotation for this pass.")
                    continue

                angc = np.arctan2(vic, uic)*180/np.pi # Angle relative to true east.
                xcperp, ycperp = get_xtrackline_from_angle(lon[fc_adt], lat[fc_adt], angc, L=Lx, dL=dx_approx)

            xyperp = (xcperp, ycperp)
            ui2 = griddata(xyADT, uiADTa, xyperp, method="linear")
            vi2 = griddata(xyADT, viADTa, xyperp, method="linear")
            if np.isnan(ui2).all() or np.isnan(vi2).all():
                print("All-NaN interpolated (u, v). Skipping coordinate rotation for this pass.")
                continue

            vi2r, ui2r = rot_vec(ui2, vi2, angle=angc)
            fc_adt2 = np.nanargmax(ui2**2 + vi2**2) # Find the origin again after interpolating to L4 ADT line. It may or may not intersect the L2 track.

            # For the L4 ADT, recalculate cross-stream axis and interpolate to x0
            # (instead of just shifting the origin) to avoid minor [O(20 m)]
            # differences due to the changing instantaneous axis orientation.
            xc_adt = np.append(0, np.cumsum(distance(xcperp, ycperp)))*1e-3
            xc_adt = xc_adt - xc_adt[nxh]
            fgu, fgv = np.isfinite(ui2r), np.isfinite(vi2r)
            viADTi = np.interp(x0, xc_adt[fgu], vi2r[fgu], left=np.nan, right=np.nan)
            uiADTi = np.interp(x0, xc_adt[fgv], ui2r[fgv], left=np.nan, right=np.nan)
            ax.plot(xcperp, ycperp, "gray") # Separate cross-stream line for L4 ADT.

        ax.plot(lon[fc_trk], lat[fc_trk], marker="+", ms=10, mfc="m", mec="m")
        ax.plot(lon[fc_adt], lat[fc_adt], marker="+", ms=10, mfc="gray", mec="gray")
        ax.text(0.01, 0.95, "Rotation angle = %1.1f$^\circ$"%angr, fontsize=12, transform=ax.transAxes)
        ax.set_title("(Cycle %d) : "%cyc + str(ti).replace("T", " ").split(".")[0][:-3])
        fig.savefig("%s/all_profiles_rotateADT_TPJAOS/cycle%.3d.png"%(namedir, cyc), bbox_inches="tight", dpi=150)
        plt.close(fig)

    xc = x0 - x0[fc]

    if PLT_INDIVIDUAL_PROFILES:
        dslai = 0.05
        fig, ax = plt.subplots()
        ax.plot(xc, slai, color="k", label="Signal")
        ax.plot(xc, slai_fit_low, color="b", alpha=0.2, label="Fit (degree %d)"%poly_deg_low)
        ax.plot(xc, slai_fit, color="r", alpha=0.2, label="Fit (degree %d)"%poly_deg)
        ax.plot(xc, slai_fit_aux, color="r", linestyle="dashed")
        ax.set_ylim(np.nanmin(slai) - dslai, np.nanmax(slai) + dslai)
        ax.axvline(x=0, color="gray", linestyle="dashed")
        ax.legend()
        ax.set_xlabel("Cross-stream distance [km]")
        ax.set_ylabel("Stream-coordinate ADT [m]")
        ax.set_title("Cycle %d"%cyc)
        fig.savefig("%s/all_profiles_TPJAOS/cycle%.3d.png"%(namedir, cyc), bbox_inches="tight")
        plt.close(fig)

    if PLT_SPAGHETTI_PROFILES:
        ax0.plot(x0, slai)

    # if cyc==46:
    #     stop
    # For the L2 ADT, just shift the origin, no need to interpolate.
    nshft = nxh - fc
    slai = rolltrim(slai, nshft)

    if slas is None:
        slas = slai[np.newaxis, :]
        times = Timestamp(ti)
        if ROTATE_USING_ADT:
           angrs = angr
           fcs_trk, fcs_adt = fc_trk, fc_adt
        if GET_L4_ADT_STREAM_VELOCITY:
            uiADTis, viADTis = uiADTi, viADTi
    else:
        slas = np.vstack((slas, slai[np.newaxis, :]))
        times = np.append(times, Timestamp(ti))
        if ROTATE_USING_ADT:
           angrs = np.append(angrs, angr)
           fcs_trk = np.append(fcs_trk, fc_trk)
           fcs_adt = np.append(fcs_adt, fc_adt)
        if GET_L4_ADT_STREAM_VELOCITY:
            uiADTis = np.vstack((uiADTis, uiADTi[np.newaxis, :]))
            viADTis = np.vstack((viADTis, viADTi[np.newaxis, :]))

if PLT_SPAGHETTI_PROFILES:
    ax0.axvline(x=0, color="gray", linestyle="dashed")
    fig0.savefig("%s/all_sla_profiles_spaghetti.png"%namedir, bbox_inches="tight", dpi=150)

if GET_L4_ADT_STREAM_VELOCITY:
    uiADTisavg = np.nanmean(uiADTis, axis=0)
    viADTisavg = np.nanmean(viADTis, axis=0)
    fig, ax = plt.subplots()
    ax.plot(x0, uiADTisavg, "b", label="$u^\parallel$")
    ax.plot(x0, viADTisavg, "r", label="$v^\perp$")
    ax.set_xlim(round(x0[0]), round(x0[-1]))
    ax.axvline(x=0, color="gray", linestyle="dashed")
    ax.axhline(y=0, color="gray", linestyle="dashed")
    ax.legend()
    ax.set_ylabel("Surface absolute $u_g$ [m/s]", fontweight="black")
    ax.set_xlabel("Cross-stream distance [km]", fontweight="black")
    ax.set_title("Geostrophic velocities from the L4 CMEMS ADT", fontweight="black")
    fig.savefig("%s/ugvg_streamavg_L4adt_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

# Plot time series of instantaneous distance between origin
# derived from L4 ADT and along-track ADT.
d0_adt_trk = x0[fcs_adt] - x0[fcs_trk]
fbadd0 = np.abs(d0_adt_trk)>=maxd0_thresh

print("")
print("Removed %d cycles with departures greater than %d km"%(fbadd0.sum(), maxd0_thresh))

fig, ax = plt.subplots()
ax.plot(times, d0_adt_trk)
ax.plot(times[fbadd0], d0_adt_trk[fbadd0], linestyle="none", marker=".", mfc="r", mec="r")
ax.set_xlim(times[0], times[-1])
ax.xaxis_date()
ax.axhline(linestyle="dashed", color="gray")
ax.text(0.01, 0.05, "Rejected points (|departure| > %d km): %d [%1.1f%%]"%(maxd0_thresh, fbadd0.sum(), 100*fbadd0.sum()/fbadd0.size), color="k", transform=ax.transAxes, zorder=999)
ax.text(0.01, 0.95, "Mean |departure| = %.1f km"%np.nanmean(np.abs(d0_adt_trk)), transform=ax.transAxes, zorder=999)
ax.set_ylabel("Distance between origins [km]", fontsize=15)
fig.savefig("%s/ug_streamavg_origin_departure_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

# Rotate to instantaneous jet frame of reference.
if ROTATE_USING_ADT:
    angrs_aux = angrs.copy()
    angrs_aux[angrs_aux>+90] -= 180
    angrs_aux[angrs_aux<-90] += 180

    fbadang = np.logical_or(angrs_aux>=maxangr_thresh, angrs_aux<=-maxangr_thresh)

    fig, ax = plt.subplots()
    ax.plot(times, angrs)
    ax.plot(times[fbadang], angrs[fbadang], linestyle="none", marker=".", mfc="r", mec="r")
    ax.set_xlim(times[0], times[-1])
    ax.xaxis_date()
    ax.axhline(linestyle="dashed", color="gray")
    ax.text(0.01, 0.05, "Rejected points ($\pm 90^\circ$ wrapped angle > $\pm$%d$^\circ$): %d [%1.1f%%]"%(maxangr_thresh, fbadang.sum(), 100*fbadang.sum()/fbadang.size), color="k", transform=ax.transAxes, zorder=999)

    print("")
    print("Removed %d cycles with rotation angles greater than %d degrees"%(fbadang.sum(), maxangr_thresh))
    angrs[fbadang] = np.nan

    angrs_aux[fbadang] = np.nan
    ax.text(0.01, 0.95, "Mean rotation angle ($\pm 90^\circ$ wrapped): %1.1f$^\circ$"%circmean(angrs_aux, high=90, low=-90, nan_policy="omit"), transform=ax.transAxes, zorder=999)
    ax.set_ylabel("Rotation angle [degrees]", fontsize=15)
    fig.savefig("%s/ug_streamavg_rotation_angle_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

slasr = slas/np.cos(np.radians(angrs[:, np.newaxis]))
detadxir = np.gradient(slasr, axis=1, edge_order=2)/dx
usir = -detadxir*ugfac

# Add left and right surface Ld from LG20 to stream average plot****
dd = loadmat("../data/misc/Ld_LaCasce-Groeskamp_2020.mat")
lonLd, latLd, Ldflati, Ldsurfi = dd["xt"].squeeze(), dd["yt"].squeeze(), dd["Ld_flat_RK4"].squeeze().T*1e-3, dd["Ld_rough_RK4"].squeeze().T*1e-3
xiLd, yiLd = np.meshgrid(lonLd, latLd)

nlr = round(Llr/dx_approx)
lonl, lonr = lon180to360(lon[nxh-nlr])[0], lon180to360(lon[nxh+nlr])[0]
latl, latr = lat[nxh-nlr], lat[nxh+nlr]

fg = np.isfinite(Ldflati)
mskflati = ~fg
Ldflatl, Ldflatr = griddata((xiLd[fg], yiLd[fg]), Ldflati[fg], ([lonl, lonr], [latl, latr]), method="cubic")
Ldflat = griddata((xiLd[fg], yiLd[fg]), Ldflati[fg], (lon180to360(lon), lat), method="cubic")
fg = np.isfinite(Ldsurfi)
msksurfi = ~fg
Ldsurfl, Ldsurfr = griddata((xiLd[fg], yiLd[fg]), Ldsurfi[fg], ([lonl, lonr], [latl, latr]), method="cubic")
Ldsurf = griddata((xiLd[fg], yiLd[fg]), Ldsurfi[fg], (lon180to360(lon), lat), method="cubic")
Ldflati[mskflati] = np.nan
Ldsurfi[msksurfi] = np.nan

Ldflat[Ldflat<Ldmin] = np.nan
Ldsurf[Ldsurf<Ldmin] = np.nan
impossibleLdsurfflat = Ldflat>Ldsurf
Ldflat[impossibleLdsurfflat] = np.nan
Ldsurf[impossibleLdsurfflat] = np.nan

pltmaps_Ld(lon, lat, sup, name, nlr, Ldflatl, Ldflatr, Ldsurfl, Ldsurfr, SAVEFIG=True)

# Remove eDOFs from all positions for rejected transects.
slas[fbadang, :] = np.nan
slas[fbadd0, :] = np.nan
slasr[fbadang, :] = np.nan
slasr[fbadd0, :] = np.nan
usir[fbadang, :] = np.nan
usir[fbadd0, :] = np.nan
fbad_usthresh = np.abs(usir)>us_thresh
slas[fbad_usthresh] = np.nan
slasr[fbad_usthresh] = np.nan
usir[fbad_usthresh] = np.nan

uiADTis[fbadang, :] = np.nan
uiADTis[fbadd0, :] = np.nan
viADTis[fbadang, :] = np.nan
viADTis[fbadd0, :] = np.nan

slaavgr = np.nanmean(slasr, axis=0)
usavgr = np.nanmean(usir, axis=0)

xEDOF = np.float32(np.isfinite(usir).sum(axis=0))
SE = np.nanstd(usir, axis=0)/np.sqrt(xEDOF)
fcavg = np.nanargmax(np.abs(usavgr))

CL95l = usavgr - 2*SE
CL95u = usavgr + 2*SE

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
ax1, ax2 = ax
ax1.plot(x0, slaavgr, "b", marker=".")
ax2.plot(x0, usavgr, "r", marker=".")
ax1.set_xlim(round(x0[0]), round(x0[-1]))
ax1.axvline(x=0, color="gray", linestyle="dashed")
ax2.axvline(x=0, color="gray", linestyle="dashed")
ax2.axhline(y=0, color="gray", linestyle="dashed")
ax1.set_ylabel("Stream-coordinate ADT [m]", fontweight="black")
ax2.set_ylabel("Surface absolute $u_g$ [m/s]", fontweight="black")
ax2.set_xlabel("Cross-stream distance [km]", fontweight="black")
fig.subplots_adjust(hspace=0)
fig.savefig("%s/adt_ug_streamavg_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

dyl, dyt = 0.1, 0.1
fig, ax = plt.subplots()
if GET_L4_ADT_STREAM_VELOCITY:
    ax.plot(x0, viADTisavg, "m", alpha=0.2)
ax.fill_between(x0, CL95l, CL95u, color="k", alpha=0.2)
ax.plot(x0, usavgr, "k", marker="o", ms=1)
ax.set_xlim(round(x0[0]), round(x0[-1]))
ax.set_ylim(np.nanmin(usavgr) - dyl, np.nanmax(usavgr) + dyl)
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.axhline(y=0, color="gray", linestyle="dashed")
ax.text(0.01, 0.95, "Max/min eDOF = %d / %d"%(np.nanmax(xEDOF), np.nanmin(xEDOF)), fontsize=12, transform=ax.transAxes)
ax.text(0.2, 0.75+dyt, "$L_{dl}$ = %.1f, %.1f km"%(Ldsurfl, Ldflatl), fontsize=12, transform=ax.transAxes, ha="center", color="r")
ax.text(0.8, 0.75+dyt, "$L_{dr}$ = %.1f, %.1f km"%(Ldsurfr, Ldflatr), fontsize=12, transform=ax.transAxes, ha="center", color="g")
ax.set_ylabel("Surface absolute geostrophic velocity [m/s]", fontsize=10, fontweight="black")
ax.set_xlabel("Cross-stream distance [km]", fontsize=14, fontweight="black")
ax.set_title(sup, fontsize=15, fontweight="black")
fig.savefig("%s/ug_streamavg_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

timesc = IndexVariable("t", times, attrs=dict(long_name="Time"))
x0c = IndexVariable("x", x0, attrs=dict(units="km", long_name="Cross-stream distance"))
coords = dict(t=timesc, x=x0c)
coordsx = dict(x=x0)
coordst = dict(t=times)

Us = DataArray(usir, coords=coords, attrs=dict(units="m/s", long_name="Along-stream surface absolute geostrophic velocity from TPJAOS L2 along-track ADT (rotated following instantaneous jet orientation)"))
Slasr = DataArray(slasr, coords=coords, attrs=dict(units="m", long_name="Cross-stream ADT from TPJAOS L2 along-track ADT (rotated following instantaneous jet orientation)"))
Slas = DataArray(slas, coords=coords, attrs=dict(units="m", long_name="Cross-stream ADT from TPJAOS L2 along-track ADT"))

# Plot annual averages (full years only, 1993-2020).
us_aux = Us.sel(dict(t=slice("1993-01-01 00:00:00", "2021-01-01 00:00:00")))
us_annavg = us_aux.resample(dict(t="1y"), loffset="-6m").mean("t")
nyears = us_annavg["t"].size

# Plot instantaneous profiles and annual averages.
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
ax1, ax2 = ax
Us.plot(ax=ax1)
us_annavg.plot(ax=ax2)
ax1.axvline(x=0, color="k", linestyle="dashed")
ax2.axvline(x=0, color="k", linestyle="dashed")
fig.subplots_adjust(hspace=0)
fig.savefig("%s/ug_streamavg_annual_heatmap_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

fig, ax = plt.subplots()
ax.set_xlim(round(x0[0]), round(x0[-1]))
for n in range(nyears):
    ax.plot(x0, us_annavg.isel(dict(t=n)), marker="o", ms=1)
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.axhline(y=0, color="gray", linestyle="dashed")
ax.set_ylabel("Surface absolute geostrophic velocity [m/s]", fontsize=10, fontweight="black")
ax.set_xlabel("Cross-stream distance [km]", fontsize=14, fontweight="black")
ax.set_title(sup + " annual averages", fontsize=13, fontweight="black")
fig.savefig("%s/ug_streamavg_annual_spaghetti_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

# Save output.
trklon = DataArray(xyalongtrk[0], coords=coordsx, attrs=dict(units="Degrees East", long_name="Longitude of ground track"))
trklat = DataArray(xyalongtrk[1], coords=coordsx, attrs=dict(units="Degrees North", long_name="Latitude of ground track"))
xEDOF = DataArray(xEDOF, coords=coordsx, attrs=dict(units="unitless", long_name="Number of effective degrees of freedom based on valid along-track ADT observations"))
angrs = DataArray(angrs, coords=coordst, attrs=dict(units="degrees", long_name="Rotation angle between altimeter track and instantaneous stream orientation"))
d0_adt_trk = DataArray(d0_adt_trk, coords=coordst, attrs=dict(units="km", long_name="Departure between TPJAOS along-track origin and L4 CMEMS ADT origin"))
uiADTis = DataArray(uiADTis, coords=coords, attrs=dict(units="m/s", long_name="Cross-stream velocity derived from 2D L4 ADT CMEMS maps"))
viADTis = DataArray(viADTis, coords=coords, attrs=dict(units="m/s", long_name="Along-stream velocity derived from 2D L4 ADT CMEMS maps"))
Ldflat = DataArray(Ldflat, coords=coordsx, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf = DataArray(Ldsurf, coords=coordsx, attrs=dict(units="km", long_name="First surface deformation radius"))

fc0 = int(np.median(fcs_trk)) # Along-track index in the (lon, lat) arrays where the jet core is most often. If fc0 = nxh, it is the center of the original cross-stream distance array.
fxshift = fc0 - nxh
Ldflat = Ldflat.shift(dict(x=fxshift))
Ldsurf = Ldsurf.shift(dict(x=fxshift))

dxshift = x0[fc0] - x0[nxh]
print("")
print("Ld cross-stream profiles have been shifted by %d indices (%1.1f km)"%(fxshift, dxshift))

dvars = dict(us=Us, slas=Slasr, slas_nonrotated=Slas, lon=trklon, lat=trklat, xEDOF=xEDOF, Ldflat=Ldflat, Ldsurf=Ldsurf, angr=angrs, dx0trk=d0_adt_trk, uADTs=uiADTis, vADTs=viADTis)
attrs = dict(rotation_angle_note="Cycles with rotation angle (wrapped between +-90 degrees) greater than +-%d degrees were masked out"%maxangr_thresh, origin_departure_note="Cycles where the departure between the origin determined from the L4 ADT and the origin determined from the along-track ADT greater than %d km were masked out"%maxd0_thresh)
dsout = Dataset(data_vars=dvars, coords=coords, attrs=attrs)

# if name in CURRENTS_FLIP:
Ldlavg = Ldsurf.where(dsout["x"]<0).mean().values.flatten()[0]
Ldravg = Ldsurf.where(dsout["x"]>0).mean().values.flatten()[0]
if Ldlavg>Ldravg:
    dsout["x"] = - dsout["x"]
    dsout = dsout.sortby("x")

dsout.to_netcdf("%s/ug_stream_TPJAOS-%s.nc"%(namedir, name))


fig, ax = plt.subplots()
Ldflat.plot(ax=ax, color="b", label="Flat-bottom")
Ldsurf.plot(ax=ax, color="r", label="Surface")
ax.axvline(color="gray")
ax.set_xlim(round(x0[0]), round(x0[-1]))
ax.set_ylabel("Cross-stream distance [km]")
ax.set_ylabel("Deformation radius [km]")
ax.legend()
fig.savefig("%s/Ldprofile_LG20_TPJAOS_%s.png"%(namedir, sup.replace(" ", "_")), bbox_inches="tight", dpi=150)

plt.show(block=False)
