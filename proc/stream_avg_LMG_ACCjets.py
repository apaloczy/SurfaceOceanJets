# Calculate stream-averaged velocity for ACC jets
# observed in shipboard ADCP data (Laurence M. Gould).
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xarray import open_dataset, DataArray, Dataset, IndexVariable
from hdf5storage import loadmat
from pandas import read_table, DataFrame, Timestamp
from gsw import distance
from cmocean.cm import balance, phase
from ap_tools.utils import near, lon180to360, compass2trig, rot_vec
from glob import glob
from pygeodesy.sphericalNvector import LatLon as LatLon_sphere
from scipy.interpolate import griddata, interp1d
from ap_tools.utils import lon360to180, lon180to360, near2
import cartopy.crs as ccrs
from cartopy.feature import LAND
from shapely.geometry import LineString, Point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
warnings.filterwarnings("ignore")


def interp_sec(xT, zT, T, xiT, msktopo=True):
    nziT = zT.size
    nxiT = xiT.size
    nxiTh = nxiT//2
    Ti = np.empty((nziT, nxiT))*np.nan
    for k in range(nziT):
        fgi = np.isfinite(T[k, :])
        if fgi.any():
            Ti[k, :] = np.interp(xiT, xT[fgi], T[k, fgi], left=np.nan, right=np.nan)

    if msktopo:
        msktopo_aux = np.isnan(T)
        nxT = T.shape[1]
        botT = np.empty(nxT)*np.nan
        zTbot = zT.min()
        for i in range(nxT):
            fdeep = np.where(msktopo_aux[:, i])[0]
            if len(fdeep)>0:
                botT[i] = zT[fdeep[0]] # Depth of first NaN from surface.
            else:
                botT[i] = zTbot

        fg = np.isfinite(botT)
        botTi = np.interp(xiT, xT[fg], botT[fg], left=np.nan, right=np.nan)
        zTi = np.tile(zT[:, np.newaxis], (1, nxiT))
        fbot = zTi<botTi
        Ti[fbot] = np.nan

    return Ti


def pltmaps_Ld(lonc, latc, lonl, latl, lonr, latr, name, Ldflatl, Ldflatr, Ldsurfl, Ldsurfr, dxy=5, SAVEFIG=True):
    global xiLd, yiLd, Ldflati, Ldsurfi, xt, yt, zt, isobs

    ext = (lonc - dxy, lonc + dxy, latc - dxy, latc + dxy)
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
    ax1.plot(lon360to180(lonc), latc, marker="*", ms=6, mfc="m", mec="m", mew=0.5, zorder=10)
    ax1.plot(lon360to180(lonl), latl, marker="o", ms=6, mfc="r", mec="w", mew=0.5, zorder=10)
    ax1.plot(lon360to180(lonr), latr, marker="o", ms=6, mfc="g", mec="w", mew=0.5, zorder=10)
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
    ax2.plot(lon360to180(lonc), latc, marker="*", ms=6, mfc="m", mec="m", mew=0.5, zorder=10)
    ax2.plot(lon360to180(lonl), latl, marker="o", ms=6, mfc="r", mec="w", mew=0.5, zorder=10)
    ax2.plot(lon360to180(lonr), latr, marker="o", ms=6, mfc="g", mec="w", mew=0.5, zorder=10)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax2.set_title(" $L_{dl}, L_{dr}$ = %.1f km, %.1f km"%(Ldsurfl, Ldsurfr), fontsize=10, fontweight="black")
    cax = ax2.inset_axes((0, -0.15, 1, 0.05))
    cb = fig.colorbar(cs2, cax=cax, orientation="horizontal")
    cb.set_label("Surface $L_d$ [km]", fontsize=12, fontweight="black")
    fig.suptitle(name.replace("_", " "), fontsize=13, fontweight="black", y=0.875)
    if SAVEFIG:
        fig.savefig("ACCjets/Ld_map_%s.png"%name, bbox_inches="tight", dpi=200)


def get_closest_point(Fxing, xF, yF):
    if Fxing.geom_type=="MultiPoint":
        ps = []
        for p in Fxing:
            ps.append(p)
        Fxing = ps.copy()
        dists = []
        for p in Fxing:
            x0, y0 = p.x, p.y
            xn, yn = near2(xF, yF, x0, y0)
            dists.append(distance([x0, xn], [y0, yn])[0])
        Fxing = Fxing[np.argmin(dists)]
    elif Fxing.geom_type=="LineString":
        print("LineString set found, skipping. This most likely means that this front was not crossed by the ship track. Check on the ship track maps.")
        return None
    elif Fxing.geom_type=="Point":
        pass

    return Fxing


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
    ang = compass2trig(pt1.bearingTo(pt2))[0] - 90

    return ang


#---
plt.close("all")

SPLIT_INOUT = True
ROTATE_UV_USING_ADT = True

PLT_MAP = True
PLT_SYNOPTIC_SECTIONS = True
INTERACTIVE_SELECT_INOUT_POINTS = False

head = "../data/ADCP_LMG/"
head_XBT = "../data/XBT_LMG/"
fnameADT = "../data/ADT_CMEMS/CMEMS-ADT-DrakePassage_ACC.nc"

wanted_radius = 100 # [km]
angug_radius = 20   # [km]

# Zoom in on Drake Passage.
xmin, xmax = -80.0, -41.0
ymin, ymax = -65.0, -54.5
badTflag = -0.999
proj = ccrs.PlateCarree()

zavg_top = 50 # [m]
umax = 1.0
minpts = 4
Ls = 150 # [km]
dxs = 1.0

xs = np.linspace(-Ls, 0, num=(round(Ls/dxs)+1))
xs = np.append(xs, -np.flipud(xs[:-1]))


# Overlay climatological ACC front positions from CMEMS MDT.
f_climfronts = "../data/misc/ACC_fronts_PHI19.nc"
dsf = open_dataset(f_climfronts)
loncSACCF, latcSACCF = dsf["LonSACCF"].values, dsf["LatSACCF"].values
loncPF, latcPF = dsf["LonPF"].values, dsf["LatPF"].values
loncSAF, latcSAF = dsf["LonSAF"].values, dsf["LatSAF"].values
fg = np.logical_and(np.isfinite(loncSACCF), np.isfinite(latcSACCF))
loncSACCF, latcSACCF = loncSACCF[fg], latcSACCF[fg]
SACCFxy = LineString(np.column_stack((loncSACCF, latcSACCF)))
fg = np.logical_and(np.isfinite(loncPF), np.isfinite(latcPF))
loncPF, latcPF = loncPF[fg], latcPF[fg]
PFxy = LineString(np.column_stack((loncPF, latcPF)))
fg = np.logical_and(np.isfinite(loncSAF), np.isfinite(latcSAF))
loncSAF, latcSAF = loncSAF[fg], latcSAF[fg]
SAFxy = LineString(np.column_stack((loncSAF, latcSAF)))

finSAF_plt = np.logical_and(np.logical_and(latcSAF>=ymin, latcSAF<=ymax), np.logical_and(loncSAF>=xmin, loncSAF<=xmax))
finPF_plt = np.logical_and(np.logical_and(latcPF>=ymin, latcPF<=ymax), np.logical_and(loncPF>=xmin, loncPF<=xmax))
finSACCF_plt = np.logical_and(np.logical_and(latcSACCF>=ymin, latcSACCF<=ymax), np.logical_and(loncSACCF>=xmin, loncSACCF<=xmax))
loncSAF_plt, latcSAF_plt = loncSAF[finSAF_plt], latcSAF[finSAF_plt]
loncPF_plt, latcPF_plt = loncPF[finPF_plt], latcPF[finPF_plt]
loncSACCF_plt, latcSACCF_plt = loncSACCF[finSACCF_plt], latcSACCF[finSACCF_plt]

fnames_XBT = glob(head_XBT + "*.nc")
fnames_XBT.sort()

TccXBT = np.concatenate(([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5], np.arange(1, 15)))
tsXBT, teXBT = [], []
for f in fnames_XBT:
    tiXBT = open_dataset(f)["t"].values
    tsXBT.append(tiXBT[0])
    teXBT.append(tiXBT[-1])

tsXBT, teXBT = map(np.array, (tsXBT, teXBT))

fnames = glob(head + "*_short.nc")
fnames.sort()
nf = len(fnames)

cblab = "$u$ east [m/s]"
if INTERACTIVE_SELECT_INOUT_POINTS:
    from sys import exit

fname_split = "../data/misc/ACCjets_inout_times.txt"
if SPLIT_INOUT:
    tinout = read_table(fname_split, delimiter=" ", header=0, index_col=0, parse_dates=True).to_dict()
    toutls, toutrs = tinout["outts"], tinout["outte"]
    tinls, tinrs = tinout["ints"], tinout["inte"]

# Read in list of cruises to skip.
with open("../data/misc/bad_cruises_ACC.txt", mode="r") as f:
    aux = f.readlines()
    bad_occupations = [l.strip("\n") for l in aux]

adt = open_dataset(fnameADT)["adt"]
if ROTATE_UV_USING_ADT:
    dlladt = 1
    Lxstream = Ls
    dLxstream = dxs
    deglat2km = 1.852*60
    adtlon0, adtlat0 = adt["longitude"].values, adt["latitude"].values
    adtlon, adtlat = np.meshgrid(adtlon0, adtlat0)
    dyadt = np.gradient(adtlat, axis=0)*deglat2km
    dxadt = np.gradient(adtlon, axis=1)*np.cos(adtlat*np.pi/180)*deglat2km

TiauxsSAF, TiauxsPF, TiauxsSACCF = None, None, None
all_lats = np.array([])
n = 0
for f in fnames:
    fn = f.split("/")[-1][:5]
    print("")
    print(fn)
    ds = open_dataset(f).set_coords(("lat", "lon"))
    lon, lat, time = ds["lon"].values, ds["lat"].values, ds["time"].values
    hdng = ds["heading"].values
    fbb = np.logical_and(np.logical_and(lon>=xmin, lon<=xmax), np.logical_and(lat>=ymin, lat<=ymax))
    lon, lat, time, hdng = lon[fbb], lat[fbb], time[fbb], hdng[fbb]

    # Plot heading within the shelfbreak lat-lon box
    # to help separate outbound and inbound parts of the cruise.
    hdng = lon180to360(hdng)
    if False:
        figh, axh = plt.subplots()
        axh.plot(time, compass2trig(hdng), marker="o", ms=1)
        axh.set_ylabel("Heading [Degrees trigonometric]")
        # plt.show(block=False)
        figh.savefig("ACCjets/hdng/%s_hdng.png"%fn, bbox_inches="tight", dpi=150)
        plt.close(figh)
        continue

    # Plot track to separate outbound and inbound parts of the cruise.
    if True:
        if SPLIT_INOUT:
            fni = int(fn)
            toutl, toutr = toutls[fni], toutrs[fni]
            tinl, tinr =  tinls[fni], tinrs[fni]
            toutl, toutr, tinl, tinr = map(Timestamp, (toutl, toutr, tinl, tinr))
            fout = np.logical_and(time>=toutl, time<=toutr)
            fin = np.logical_and(time>=tinl, time<=tinr)
            lonout, latout = lon[fout], lat[fout]
            lonin, latin = lon[fin], lat[fin]
        else:
            wanted_lat = -55
            nh = lat.size//2
            idx0_out = near(lat[:nh], wanted_lat, return_index=True)
            idx0_in = near(lat[nh:], wanted_lat, return_index=True) + nh
            t0str_out = str(Timestamp(time[idx0_out])).replace(" ", "T").split(".")[0]
            t0str_in = str(Timestamp(time[idx0_in])).replace(" ", "T").split(".")[0]

        figll, (axll1, axll2) = plt.subplots(ncols=2, figsize=(12, 5))
        axll2r = axll2.twinx()
        axll1.plot(loncSAF_plt, latcSAF_plt, "gray", zorder=0)
        axll1.plot(loncPF_plt, latcPF_plt, "gray", zorder=0)
        axll1.plot(loncSACCF_plt, latcSACCF_plt, "gray", zorder=0)
        axll1.plot(lon, lat, "k", alpha=0.3, zorder=1)
        if SPLIT_INOUT:
            axll1.plot(lonout, latout, "r", linewidth=0.5, zorder=2) # Outbound.
            axll1.plot(lonin, latin, "b", linewidth=0.5, zorder=2) # Inbound.
        axll1.set_xlabel("Longitude [Degrees East]")
        axll1.set_ylabel("Latitude [Degrees North]")
        axll2.plot(time, lon, "y", label="Longitude")
        axll2r.plot(time, lat, "g", label="Latitude")
        axll2.set_xlim(time[0], time[-1])
        axll2r.set_xlim(time[0], time[-1])
        axll2.grid(axis="both"); axll2r.grid(axis="x")
        if SPLIT_INOUT:
            axll2.axvline(toutl, color="r", linestyle="dashed")
            axll2.axvline(toutr, color="r", linestyle="dashed")
            axll2.axvline(tinl, color="b", linestyle="dashed")
            axll2.axvline(tinr, color="b", linestyle="dashed")
        axll2.xaxis.set_major_locator(plt.MaxNLocator(20))
        axll2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        axll2.xaxis.set_tick_params(rotation=90)
        axll2.set_ylabel("Longitude [Degrees East]")
        axll2r.set_ylabel("Latitude [Degrees North]")
        axll1.set_title("Cruise: %s"%fn)
        axll2.set_title("Longitude (yellow), latitude (green), year: %d"%Timestamp(time[0]).year)
        figll.savefig("ACCjets/inout_sections/%s_llinou.png"%fn, bbox_inches="tight", dpi=150)
        if not SPLIT_INOUT:
            print("")
            # print("Clip out/in timestamps: %s %s  %s"%(fn, t0str_out, t0str_in))
            print("Click on the desired south ends of the out/inbound transects")
            pts = figll.ginput(n=2, timeout=0)
            tidxsn = []
            for pt in pts:
                tidxsn.append(near2(lon, lat, pt[0], pt[1], return_index=True)[0])
            tlaux, traux = Timestamp(time[tidxsn[0]]), Timestamp(time[tidxsn[1]])
            if tlaux>traux:
                tlaux, traux = traux, tlaux
            tlaux, traux = map(str, (tlaux, traux))
            tlaux, traux = tlaux.replace(" ", "T").split(".")[0], traux.replace(" ", "T").split(".")[0]
            append_tstr = "%s %s %s %s %s"%(fn, t0str_out, tlaux, traux, t0str_in)
            print("")
            print("Append the following line to file %s"%fname_split)
            print(append_tstr)
            DataFrame([append_tstr]).to_clipboard(index=False,header=False)
            print("")

        if INTERACTIVE_SELECT_INOUT_POINTS:
            plt.show(block=False); exit()
        plt.close(figll)

    if SPLIT_INOUT: #--- Get along-front velocity for the outbound and inbound legs separately.
        dso = ds.sel(dict(time=slice(toutl, toutr)))
        dsi = ds.sel(dict(time=slice(tinl, tinr)))
        dslegs = (dso, dsi)
        sprs = ("Outbound leg........", "Inbound leg........")
    else:
        dslegs = (ds)
        sprs = ("Outbound/inbound legs together........")

    for ds, spr in zip(dslegs, sprs):
        iostr = spr.split(" ")[0]
        if iostr=="Outbound":
            iostr_short = "out"
        elif iostr=="Inbound":
            iostr_short = "in"
        elif iostr=="Outbound/inbound":
            iostr_short = "inout"

        if fn + " " + iostr_short in bad_occupations:
            print(fn + " " + iostr_short + " is on the bad occupations list. Skipping.....")
            continue

        print("")
        print(spr)
        lon, lat, time = ds["lon"].values, ds["lat"].values, ds["time"].values
        hdng = ds["heading"].values
        ts, te = str(Timestamp(time.min())).split(" ")[0], str(Timestamp(time.max())).split(" ")[0]
        tsstr, testr = Timestamp(time.min()).strftime("%Y-%m-%d %H:%Mh"), Timestamp(time.max()).strftime("%Y-%m-%d %H:%Mh")
        z = -ds["depth"].values
        u, v = ds["u"].values, ds["v"].values
        ntopbins_avg = np.sum(np.nanmedian(z, axis=0)>=-zavg_top)

        fg = np.logical_and(np.isfinite(lon), np.isfinite(lat))
        lon, lat, z = lon[fg], lat[fg], z[fg, :]
        u, v = u[fg, :], v[fg, :]
        time = time[fg]
        hdng = hdng[fg]

        # Make sure outbound (inbound) transects are southbound (northbound) in order to concatenate cross-stream profiles with their poleward lobes to the left of the core.
        if iostr_short=="out":
            assert lat[-1]<lat[0], "Outbound transect is not southbound. There is something wrong."
        elif iostr_short=="in":
            assert lat[-1]>lat[0], "Inbound transect is not northbound. There is something wrong."

        # Find points where ship track intersects ACC fronts.
        shiptrack = LineString(np.column_stack((lon, lat)))
        SAFxing = SAFxy.intersection(shiptrack)
        PFxing = PFxy.intersection(shiptrack)
        SACCFxing = SACCFxy.intersection(shiptrack)

        # Find closest intersecting point at each front.
        SAFxing = get_closest_point(SAFxing, loncSAF, latcSAF)
        PFxing = get_closest_point(PFxing, loncPF, latcPF)
        SACCFxing = get_closest_point(SACCFxing, loncSACCF, latcSACCF)

        # Also save actual (lon, lat) of climatological front crossings for each transect, just to plot vertical lines on contour plot.
        if SAFxing is None:
            print("No SAF crossing located. Skipping SAF in this occupation.")
            DO_SAF = False
            # continue
        else:
            fSAFp = near2(lon, lat, SAFxing.x, SAFxing.y, return_index=True)[0]
            SAFlonp, SAFlatp = near2(lon, lat, SAFxing.x, SAFxing.y, return_index=False)
            DO_SAF = True

        if PFxing is None:
            print("No PF crossing located. Skipping PF in this occupation.")
            DO_PF = False
            # continue
        else:
            fPFp = near2(lon, lat, PFxing.x, PFxing.y, return_index=True)[0]
            PFlonp, PFlatp = near2(lon, lat, PFxing.x, PFxing.y, return_index=False)
            DO_PF = True

        if SACCFxing is None:
            print("No SACCF crossing located. Skipping SACCF in this occupation.")
            DO_SACCF = False
            # continue
        else:
            fSACCFp = near2(lon, lat, SACCFxing.x, SACCFxing.y, return_index=True)[0]
            SACCFlonp, SACCFlatp = near2(lon, lat, SACCFxing.x, SACCFxing.y, return_index=False)
            DO_SACCF = True

        dist = np.append(0, np.nancumsum(distance(lon, lat)))*1e-3
        if DO_SAF:
            fSAF = np.abs(dist - dist[fSAFp]) <= wanted_radius

            nptsSAF = fSAF.sum()
            if nptsSAF<minpts:
                print("Too few (%d) points located near SAF. Skipping SAF in this occupation."%nptsSAF)
                DO_SAF = False

            tSAF = time[fSAF]
            tcSAF = Timestamp(tSAF[tSAF.size//2])

        if DO_PF:
            fPF = np.abs(dist - dist[fPFp]) <= wanted_radius

            nptsPF = fPF.sum()
            if nptsPF<minpts:
                print("Too few (%d) points located near PF. Skipping PF in this occupation."%nptsPF)
                DO_PF = False

            tPF = time[fPF]
            tcPF = Timestamp(tPF[tPF.size//2])

            tPF = time[fPF]
            tcPF = Timestamp(tPF[tPF.size//2])

        if DO_SACCF:
            fSACCF = np.abs(dist - dist[fSACCFp]) <= wanted_radius

            nptsSACCF = fSACCF.sum()
            if nptsSACCF<minpts:
                print("Too few (%d) points located near SACCF. Skipping SACCF in this occupation."%nptsSACCF)
                DO_SACCF = False

            tSACCF = time[fSACCF]
            tcSACCF = Timestamp(tSACCF[tSACCF.size//2])

            tSACCF = time[fSACCF]
            tcSACCF = Timestamp(tSACCF[tSACCF.size//2])

        if DO_SAF:
            lonSAF, latSAF = lon[fSAF], lat[fSAF]
            hdngSAF = hdng[fSAF]
            xSAF0 = np.append(0, np.nancumsum(distance(lonSAF, latSAF)))*1e-3
            zSAF = z[fSAF, :]
            uSAF, vSAF = u[fSAF, :], v[fSAF, :]

            try:
                fcore = np.nanargmax(np.nanmean(uSAF[:, :ntopbins_avg], axis=1))
                loncoreSAF, latcoreSAF = lonSAF[fcore], latSAF[fcore] # Position of located jet core o plot on map with climatological fronts.
            except ValueError:
                print("Error finding maximum zonal velocity core for SAF. Skipping.")
                continue

            xSAF = xSAF0 - xSAF0[fcore]
            xpSAF = np.tile(xSAF[:, np.newaxis], (1, zSAF.shape[1]))

            # Average near-surface bins.
            uSAFsurf = np.nanmean(uSAF[:, :ntopbins_avg], axis=1)
            vSAFsurf = np.nanmean(vSAF[:, :ntopbins_avg], axis=1)

        if DO_PF:
            lonPF, latPF = lon[fPF], lat[fPF]
            hdngPF = hdng[fPF]
            xPF0 = np.append(0, np.nancumsum(distance(lonPF, latPF)))*1e-3
            zPF = z[fPF, :]
            uPF, vPF = u[fPF, :], v[fPF, :]

            try:
                fcore = np.nanargmax(np.nanmean(uPF[:, :ntopbins_avg], axis=1))
                loncorePF, latcorePF = lonPF[fcore], latPF[fcore] # Position of located jet core o plot on map with climatological fronts.
            except ValueError:
                print("Error finding maximum zonal velocity core for PF. Skipping.")
                continue

            xPF = xPF0 - xPF0[fcore]
            xpPF = np.tile(xPF[:, np.newaxis], (1, zPF.shape[1]))

            # Average near-surface bins.
            uPFsurf = np.nanmean(uPF[:, :ntopbins_avg], axis=1)
            vPFsurf = np.nanmean(vPF[:, :ntopbins_avg], axis=1)

        if DO_SACCF:
            lonSACCF, latSACCF = lon[fSACCF], lat[fSACCF]
            hdngSACCF = hdng[fSACCF]
            xSACCF0 = np.append(0, np.nancumsum(distance(lonSACCF, latSACCF)))*1e-3
            zSACCF = z[fSACCF, :]
            uSACCF, vSACCF = u[fSACCF, :], v[fSACCF, :]

            try:
                fcore = np.nanargmax(np.nanmean(uSACCF[:, :ntopbins_avg], axis=1))
                loncoreSACCF, latcoreSACCF = lonSACCF[fcore], latSACCF[fcore] # Position of located jet core o plot on map with climatological fronts.
            except ValueError:
                print("Error finding maximum zonal velocity core for SACCF. Skipping.")
                continue

            xSACCF = xSACCF0 - xSACCF0[fcore]
            xpSACCF = np.tile(xSACCF[:, np.newaxis], (1, zSACCF.shape[1]))

            # Average near-surface bins.
            uSACCFsurf = np.nanmean(uSACCF[:, :ntopbins_avg], axis=1)
            vSACCFsurf = np.nanmean(uSACCF[:, :ntopbins_avg], axis=1)

        # Get collocated XBT T transects.
        plt_SAF_XBT, plt_PF_XBT, plt_SACCF_XBT = False, False, False
        fXBT_transect = None
        for tsT, teT in zip(tsXBT, teXBT):
            fXBT = np.logical_and(tSAF>=tsT, tSAF<=teT)
            if fXBT.any():
                fXBT_transect = np.where(tsXBT==tsT)[0][0]

        if fXBT_transect is not None:
            dsXBT = open_dataset(fnames_XBT[fXBT_transect])
            tT = dsXBT["t"]

            # SAF
            if DO_SAF:
                fT = np.logical_and(tT>=tSAF[0], tT<=tSAF[-1]).values
                nXBTs = fT.sum().flatten()[0]
                if nXBTs>1:
                    plt_SAF_XBT = True
                    print("%s -> XBT profiles in SAF jet: "%fn, nXBTs)
                    lonT, latT = dsXBT["lon"].values[fT], dsXBT["lat"].values[fT]
                    xTSAF = np.append(0, np.cumsum(distance(lonT, latT)))*1e-3
                    zTSAF = dsXBT["z"].values
                    TSAF = dsXBT["T"].values[:, fT]
                    TSAF[TSAF==badTflag] = np.nan

                    # Interpolate T in stream coordinates to calculate time-averaged T section.
                    TiSAF = interp_sec(xTSAF, zTSAF, TSAF, xSAF0, msktopo=True)


            # PF
            if DO_PF:
                fT = np.logical_and(tT>=tPF[0], tT<=tPF[-1]).values
                nXBTs = fT.sum().flatten()[0]
                if nXBTs>1:
                    plt_PF_XBT = True
                    print("%s -> XBT profiles in PF jet: "%fn, nXBTs)
                    lonT, latT = dsXBT["lon"].values[fT], dsXBT["lat"].values[fT]
                    xTPF = np.append(0, np.cumsum(distance(lonT, latT)))*1e-3
                    zTPF = dsXBT["z"].values
                    TPF = dsXBT["T"].values[:, fT]
                    TPF[TPF==badTflag] = np.nan

                    # Interpolate T in stream coordinates to calculate time-averaged T section.
                    TiPF = interp_sec(xTPF, zTPF, TPF, xPF0, msktopo=True)


            # SACCF
            if DO_SACCF:
                fT = np.logical_and(tT>=tSACCF[0], tT<=tSACCF[-1]).values
                nXBTs = fT.sum().flatten()[0]
                if nXBTs>1:
                    plt_SACCF_XBT = True
                    print("%s -> XBT profiles in SACCF jet: "%fn, nXBTs)
                    lonT, latT = dsXBT["lon"].values[fT], dsXBT["lat"].values[fT]
                    xTSACCF = np.append(0, np.cumsum(distance(lonT, latT)))*1e-3
                    zTSACCF = dsXBT["z"].values
                    TSACCF = dsXBT["T"].values[:, fT]
                    TSACCF[TSACCF==badTflag] = np.nan

                    # Interpolate T in stream coordinates to calculate time-averaged T section.
                    TiSACCF = interp_sec(xTSACCF, zTSACCF, TSACCF, xSACCF0, msktopo=True)

        # Rotate velocities about SAF, PF and SACCF locations using ADT to approximate the orientation of the surface geostrophic streamlines.
        #######
        # SAF #
        #######
        if ROTATE_UV_USING_ADT:
            if DO_SAF:
                adtiSAF = adt.interp(dict(time=tcSAF), method="linear")
                dadtdy, dadtdx = np.gradient(adtiSAF.values)
                dadtdy, dadtdx = dadtdy/dyadt, dadtdx/dxadt # [m/km]
                angug = adtiSAF.copy()
                angug.values = np.arctan2(dadtdy, dadtdx)*180/np.pi - 90 # Direction (in trig convention) of the surface absolute geostrophic velocity vector.
                angug.values[angug.values<-180] += 360 # Wrap at +-180 degrees.
                angug.values[angug.values>180] -= 360
                angug.name = "Surface geostrophic velocity vector angle"
                angug.attrs = dict()
                fg = np.isfinite(angug.values)
                pts, vals = (adtlon[fg], adtlat[fg]), angug.values[fg]
                angugx = griddata(pts, vals, (lonSAF, latSAF), method="cubic")
                fcang = np.abs(xSAF) <= angug_radius
                angugxcore = griddata(pts, vals, (lonSAF[fcang], latSAF[fcang]), method="cubic")
                angugxcorecSAF = np.median(angugxcore)
                shiptrackangSAF = get_trkang(lonSAF[0], lonSAF[-1], latSAF[0], latSAF[-1])

                fig, ax = plt.subplots()
                ax.plot(xSAF, angugx, "k")
                ax.plot(xSAF[fcang], angugxcore, "r", linewidth=3)
                ax.set_xlim(xSAF[0], xSAF[-1])
                ax.axhline(y=angugxcorecSAF, color="r", linestyle="dashed")
                ax.axhline(y=shiptrackangSAF, color="b", linestyle="dashed")
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.text(0, angugxcorecSAF + 10, r"$\vec{u_g}$ angle: %d$^\degree$"%angugxcorecSAF, fontsize=16, color="r")
                ax.text(0, shiptrackangSAF - 10, r"Ship track-normal: %d$^\degree$"%shiptrackangSAF, fontsize=16, color="b", ha="right")
                ax.set_xlabel("Cross-SAF distance [km]", fontsize=15)
                ax.set_ylabel("Geostrophic streamline angle [Degrees]", fontsize=15)
                ax.set_title("[SAF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=10)
                xperp, yperp = get_xtrackline_from_angle(loncoreSAF, latcoreSAF, angugxcorecSAF, L=Lxstream, dL=dLxstream)
                fig.savefig("ACCjets/ADTangxtrk/%s_%s_SAF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.xlabels_top = False
                gl.ylabels_right = False
                angug.plot.pcolormesh(ax=ax, cmap=phase, zorder=0, vmin=-180, vmax=180)
                adtiSAF.plot.contour(ax=ax, levels=100, linestyles="solid", linewidths=0.2, colors="k", zorder=1)
                ax.set_xlim(lonSAF.min() - dlladt, lonSAF.max() + dlladt)
                ax.set_ylim(latSAF.min() - dlladt, latSAF.max() + dlladt)
                ax.plot(lon, lat, "w")
                ax.plot(lonSAF, latSAF, "k")
                ax.plot(lonSAF[fcang], latSAF[fcang], "r")
                ax.plot(xperp, yperp, "r--")
                ax.plot(loncoreSAF, latcoreSAF, marker="o", ms=5, mfc="k", mec="w")
                fig.suptitle("[SAF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=8, x=0.6)
                fig.savefig("ACCjets/ADTangmaps/%s_%s_SAF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                print("SAF ship track angle, rotation angle: %d, %d degrees"%(shiptrackangSAF, angugxcorecSAF))
                uSAFsurfr, vSAFsurfr = rot_vec(uSAFsurf, vSAFsurf, angle=angugxcorecSAF)

                fig, ax = plt.subplots()
                ax.plot(xSAF, vSAFsurf, "b-", alpha=0.2, label="Northward velocity")
                ax.plot(xSAF, vSAFsurfr, "b--", alpha=0.2, label="Cross-stream velocity")
                ax.plot(xSAF, uSAFsurf, "r-", label="Eastward velocity")
                ax.plot(xSAF, uSAFsurfr, "r--", label="Downstream velocity")
                ax.set_xlim(xSAF[0], xSAF[-1])
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.legend(frameon=False)
                ax.set_xlabel("Cross-SAF distance [km]", fontsize=15)
                ax.set_ylabel("Velocity [m/s]", fontsize=15)
                ax.set_title("[SAF] %s %s, %s $\longrightarrow$ %s, Rotation angle = %d$^\degree$"%(fn, iostr_short, tsstr, testr, angugxcorecSAF), fontsize=8)
                fig.savefig("ACCjets/ADTanguvrot/%s_%s_SAF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                uSAFsurf, vSAFsurf = uSAFsurfr.copy(), vSAFsurfr.copy() # Overwrite velocity profiles with rotated velocities to save.

            ######
            # PF #
            ######
            if DO_PF:
                adtiPF = adt.interp(dict(time=tcPF), method="linear")
                dadtdy, dadtdx = np.gradient(adtiPF.values)
                dadtdy, dadtdx = dadtdy/dyadt, dadtdx/dxadt # [m/km]
                angug = adtiPF.copy()
                angug.values = np.arctan2(dadtdy, dadtdx)*180/np.pi - 90 # Direction (in trig convention) of the surface absolute geostrophic velocity vector.
                angug.values[angug.values<-180] += 360 # Wrap at +-180 degrees.
                angug.values[angug.values>180] -= 360
                angug.name = "Surface geostrophic velocity vector angle"
                angug.attrs = dict()
                fg = np.isfinite(angug.values)
                pts, vals = (adtlon[fg], adtlat[fg]), angug.values[fg]
                angugx = griddata(pts, vals, (lonPF, latPF), method="cubic")
                fcang = np.abs(xPF) <= angug_radius
                angugxcore = griddata(pts, vals, (lonPF[fcang], latPF[fcang]), method="cubic")
                angugxcorecPF = np.median(angugxcore)
                shiptrackangPF = get_trkang(lonPF[0], lonPF[-1], latPF[0], latPF[-1])

                fig, ax = plt.subplots()
                ax.plot(xPF, angugx, "k")
                ax.plot(xPF[fcang], angugxcore, "r", linewidth=3)
                ax.set_xlim(xPF[0], xPF[-1])
                ax.axhline(y=angugxcorecPF, color="r", linestyle="dashed")
                ax.axhline(y=shiptrackangPF, color="b", linestyle="dashed")
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.text(0, angugxcorecPF + 10, r"$\vec{u_g}$ angle: %d$^\degree$"%angugxcorecPF, fontsize=16, color="r")
                ax.text(0, shiptrackangPF - 10, r"Ship track-normal: %d$^\degree$"%shiptrackangPF, fontsize=16, color="b", ha="right")
                ax.set_xlabel("Cross-PF distance [km]", fontsize=15)
                ax.set_ylabel("Geostrophic streamline angle [Degrees]", fontsize=15)
                ax.set_title("[PF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=10)
                xperp, yperp = get_xtrackline_from_angle(loncorePF, latcorePF, angugxcorecPF, L=Lxstream, dL=dLxstream)
                fig.savefig("ACCjets/ADTangxtrk/%s_%s_PF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.xlabels_top = False
                gl.ylabels_right = False
                angug.plot.pcolormesh(ax=ax, cmap=phase, zorder=0, vmin=-180, vmax=180)
                adtiPF.plot.contour(ax=ax, levels=100, linestyles="solid", linewidths=0.2, colors="k", zorder=1)
                ax.set_xlim(lonPF.min() - dlladt, lonPF.max() + dlladt)
                ax.set_ylim(latPF.min() - dlladt, latPF.max() + dlladt)
                ax.plot(lon, lat, "w")
                ax.plot(lonPF, latPF, "k")
                ax.plot(lonPF[fcang], latPF[fcang], "r")
                ax.plot(xperp, yperp, "r--")
                ax.plot(loncorePF, latcorePF, marker="o", ms=5, mfc="k", mec="w")
                fig.suptitle("[PF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=8, x=0.6)
                fig.savefig("ACCjets/ADTangmaps/%s_%s_PF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                print("PF ship track angle, rotation angle: %d, %d degrees"%(shiptrackangPF, angugxcorecPF))
                uPFsurfr, vPFsurfr = rot_vec(uPFsurf, vPFsurf, angle=angugxcorecPF)

                fig, ax = plt.subplots()
                ax.plot(xPF, vPFsurf, "b-", alpha=0.2, label="Northward velocity")
                ax.plot(xPF, vPFsurfr, "b--", alpha=0.2, label="Cross-stream velocity")
                ax.plot(xPF, uPFsurf, "r-", label="Eastward velocity")
                ax.plot(xPF, uPFsurfr, "r--", label="Downstream velocity")
                ax.set_xlim(xPF[0], xPF[-1])
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.legend(frameon=False)
                ax.set_xlabel("Cross-PF distance [km]", fontsize=15)
                ax.set_ylabel("Velocity [m/s]", fontsize=15)
                ax.set_title("[PF] %s %s, %s $\longrightarrow$ %s, Rotation angle = %d$^\degree$"%(fn, iostr_short, tsstr, testr, angugxcorecPF), fontsize=8)
                fig.savefig("ACCjets/ADTanguvrot/%s_%s_PF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                uPFsurf, vPFsurf = uPFsurfr.copy(), vPFsurfr.copy() # Overwrite velocity profiles with rotated velocities to save.


            #########
            # SACCF #
            #########
            if DO_SACCF:
                adtiSACCF = adt.interp(dict(time=tcSACCF), method="linear")
                dadtdy, dadtdx = np.gradient(adtiSACCF.values)
                dadtdy, dadtdx = dadtdy/dyadt, dadtdx/dxadt # [m/km]
                angug = adtiSACCF.copy()
                angug.values = np.arctan2(dadtdy, dadtdx)*180/np.pi - 90 # Direction (in trig convention) of the surface absolute geostrophic velocity vector.
                angug.values[angug.values<-180] += 360 # Wrap at +-180 degrees.
                angug.values[angug.values>180] -= 360
                angug.name = "Surface geostrophic velocity vector angle"
                angug.attrs = dict()
                fg = np.isfinite(angug.values)
                pts, vals = (adtlon[fg], adtlat[fg]), angug.values[fg]
                angugx = griddata(pts, vals, (lonSACCF, latSACCF), method="cubic")
                fcang = np.abs(xSACCF) <= angug_radius
                angugxcore = griddata(pts, vals, (lonSACCF[fcang], latSACCF[fcang]), method="cubic")
                angugxcorecSACCF = np.median(angugxcore)
                shiptrackangSACCF = get_trkang(lonSACCF[0], lonSACCF[-1], latSACCF[0], latSACCF[-1])

                fig, ax = plt.subplots()
                ax.plot(xSACCF, angugx, "k")
                ax.plot(xSACCF[fcang], angugxcore, "r", linewidth=3)
                ax.set_xlim(xSACCF[0], xSACCF[-1])
                ax.axhline(y=angugxcorecSACCF, color="r", linestyle="dashed")
                ax.axhline(y=shiptrackangSACCF, color="b", linestyle="dashed")
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.text(0, angugxcorecSACCF + 10, r"$\vec{u_g}$ angle: %d$^\degree$"%angugxcorecSACCF, fontsize=16, color="r")
                ax.text(0, shiptrackangSACCF - 10, r"Ship track-normal: %d$^\degree$"%shiptrackangSACCF, fontsize=16, color="b", ha="right")
                ax.set_xlabel("Cross-SACCF distance [km]", fontsize=15)
                ax.set_ylabel("Geostrophic streamline angle [Degrees]", fontsize=15)
                ax.set_title("[SACCF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=10)
                xperp, yperp = get_xtrackline_from_angle(loncoreSACCF, latcoreSACCF, angugxcorecSACCF, L=Lxstream, dL=dLxstream)
                fig.savefig("ACCjets/ADTangxtrk/%s_%s_SACCF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True, linewidth=0)
                gl.xlabels_top = False
                gl.ylabels_right = False
                angug.plot.pcolormesh(ax=ax, cmap=phase, zorder=0, vmin=-180, vmax=180)
                adtiSACCF.plot.contour(ax=ax, levels=100, linestyles="solid", linewidths=0.2, colors="k", zorder=1)
                ax.set_xlim(lonSACCF.min() - dlladt, lonSACCF.max() + dlladt)
                ax.set_ylim(latSACCF.min() - dlladt, latSACCF.max() + dlladt)
                ax.plot(lon, lat, "w")
                ax.plot(lonSACCF, latSACCF, "k")
                ax.plot(lonSACCF[fcang], latSACCF[fcang], "r")
                ax.plot(xperp, yperp, "r--")
                ax.plot(loncoreSACCF, latcoreSACCF, marker="o", ms=5, mfc="k", mec="w")
                fig.suptitle("[SACCF] %s %s, %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), fontsize=8, x=0.6)
                fig.savefig("ACCjets/ADTangmaps/%s_%s_SACCF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                print("SACCF ship track angle, rotation angle: %d, %d degrees"%(shiptrackangSACCF, angugxcorecSACCF))
                uSACCFsurfr, vSACCFsurfr = rot_vec(uSACCFsurf, vSACCFsurf, angle=angugxcorecSACCF)

                fig, ax = plt.subplots()
                ax.plot(xSACCF, vSACCFsurf, "b-", alpha=0.2, label="Northward velocity")
                ax.plot(xSACCF, vSACCFsurfr, "b--", alpha=0.2, label="Cross-stream velocity")
                ax.plot(xSACCF, uSACCFsurf, "r-", label="Eastward velocity")
                ax.plot(xSACCF, uSACCFsurfr, "r--", label="Downstream velocity")
                ax.set_xlim(xSACCF[0], xSACCF[-1])
                ax.axhline(color="gray")
                ax.axvline(color="gray")
                ax.legend(frameon=False)
                ax.set_xlabel("Cross-SACCF distance [km]", fontsize=15)
                ax.set_ylabel("Velocity [m/s]", fontsize=15)
                ax.set_title("[SACCF] %s %s, %s $\longrightarrow$ %s, Rotation angle = %d$^\degree$"%(fn, iostr_short, tsstr, testr, angugxcorecSACCF), fontsize=8)
                fig.savefig("ACCjets/ADTanguvrot/%s_%s_SACCF.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
                plt.close(fig)

                uSACCFsurf, vSACCFsurf = uSACCFsurfr.copy(), vSACCFsurfr.copy() # Overwrite velocity profiles with rotated velocities to save.

        if DO_SAF:
            uauxSAF = np.interp(xs, xSAF, uSAFsurf, left=np.nan, right=np.nan)
        if DO_PF:
            uauxPF = np.interp(xs, xPF, uPFsurf, left=np.nan, right=np.nan)
        if DO_SACCF:
            uauxSACCF = np.interp(xs, xSACCF, uSACCFsurf, left=np.nan, right=np.nan)

        # Keep the poleward side of the jets on the left (x<0) side.
        if iostr_short=="out":
            if DO_SAF:
                uauxSAF = np.flipud(uauxSAF)
            if DO_PF:
                uauxPF = np.flipud(uauxPF)
            if DO_SACCF:
                uauxSACCF = np.flipud(uauxSACCF)

        if DO_SAF:
            lonSAFaux = np.interp(xs, xSAF, lonSAF, left=np.nan, right=np.nan)
            latSAFaux = np.interp(xs, xSAF, latSAF, left=np.nan, right=np.nan)
            angdiffSAF = np.abs(shiptrackangSAF - angugxcorecSAF)
        if DO_PF:
            lonPFaux = np.interp(xs, xPF, lonPF, left=np.nan, right=np.nan)
            latPFaux = np.interp(xs, xPF, latPF, left=np.nan, right=np.nan)
            angdiffPF = np.abs(shiptrackangPF - angugxcorecPF)
        if DO_SACCF:
            lonSACCFaux = np.interp(xs, xSACCF, lonSACCF, left=np.nan, right=np.nan)
            latSACCFaux = np.interp(xs, xSACCF, latSACCF, left=np.nan, right=np.nan)
            angdiffSACCF = np.abs(shiptrackangSACCF - angugxcorecSACCF)

        # Keep the poleward flank of the jets always on the left (x<0) side.
        if iostr_short=="out":
            if DO_SAF:
                lonSAFaux, latSAFaux = map(np.flipud, (lonSAFaux, latSAFaux))
            if DO_PF:
                lonPFaux, latPFaux = map(np.flipud, (lonPFaux, latPFaux))
            if DO_SACCF:
                lonSACCFaux, latSACCFaux = map(np.flipud, (lonSACCFaux, latSACCFaux))

        if n==0:
            if DO_SAF:
                tcsSAF = tcSAF
                usSAF = uauxSAF[np.newaxis, :]
                lonsSAF, latsSAF = lonSAFaux[np.newaxis, :], latSAFaux[np.newaxis, :]
                angdiffsSAF = angdiffSAF
            if DO_PF:
                tcsPF = tcPF
                usPF = uauxPF[np.newaxis, :]
                lonsPF, latsPF = lonPFaux[np.newaxis, :], latPFaux[np.newaxis, :]
                angdiffsPF = angdiffPF
            if DO_SACCF:
                tcsSACCF = tcSACCF
                usSACCF = uauxSACCF[np.newaxis, :]
                lonsSACCF, latsSACCF = lonSACCFaux[np.newaxis, :], latSACCFaux[np.newaxis, :]
                angdiffsSACCF = angdiffSACCF
        else:
            if DO_SAF:
                tcsSAF = np.append(tcsSAF, tcSAF)
                usSAF = np.vstack((usSAF, uauxSAF[np.newaxis, :]))
                lonsSAF, latsSAF = np.vstack((lonsSAF, lonSAFaux[np.newaxis, :])), np.vstack((latsSAF, latSAFaux[np.newaxis, :]))
                angdiffsSAF = np.append(angdiffsSAF, angdiffSAF)
            if DO_PF:
                tcsPF = np.append(tcsPF, tcPF)
                usPF = np.vstack((usPF, uauxPF[np.newaxis, :]))
                lonsPF, latsPF = np.vstack((lonsPF, lonPFaux[np.newaxis, :])), np.vstack((latsPF, latPFaux[np.newaxis, :]))
                angdiffsPF = np.append(angdiffsPF, angdiffPF)
            if DO_SACCF:
                tcsSACCF = np.append(tcsSACCF, tcSACCF)
                usSACCF = np.vstack((usSACCF, uauxSACCF[np.newaxis, :]))
                lonsSACCF, latsSACCF = np.vstack((lonsSACCF, lonSACCFaux[np.newaxis, :])), np.vstack((latsSACCF, latSACCFaux[np.newaxis, :]))
                angdiffsSACCF = np.append(angdiffsSACCF, angdiffSACCF)

        # Plot ADCP velocity sections in the vicinity of each front and locate the jets.
        if PLT_SYNOPTIC_SECTIONS:
            fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
            ax1, ax2, ax3 = ax
            ax1.set_ylim(-400, 0)

            # SAF.
            if DO_SAF:
                cs = ax1.pcolormesh(xpSAF, zSAF, uSAF, vmin=-umax, vmax=umax, cmap=balance)
                if plt_SAF_XBT:
                    cc = ax1.contour(xSAF, zTSAF, TiSAF, levels=TccXBT, colors="gray", zorder=999)
                    ax1.clabel(cc)
                ax1.axvline(x=0, color="gray", linestyle="dashed")

            # PF.
            if DO_PF:
                cs = ax2.pcolormesh(xpPF, zPF, uPF, vmin=-umax, vmax=umax, cmap=balance)
                if plt_PF_XBT:
                    cc = ax2.contour(xPF, zTPF, TiPF, levels=TccXBT, colors="gray", zorder=999)
                    ax2.clabel(cc)
                ax2.axvline(x=0, color="gray", linestyle="dashed")

            # SACCF.
            if DO_SACCF:
                cs = ax3.pcolormesh(xpSACCF, zSACCF, uSACCF, vmin=-umax, vmax=umax, cmap=balance)
                if plt_SACCF_XBT:
                    cc = ax3.contour(xSACCF, zTSACCF, TiSACCF, levels=TccXBT, colors="gray", zorder=999)
                    ax3.clabel(cc)
                ax3.axvline(x=0, color="gray", linestyle="dashed")

            cax = ax1.inset_axes([0.01, 1.05, 0.25, 0.1])
            cbl = fig.colorbar(cs, cax=cax, orientation="horizontal", ticklocation="top")
            cbl.set_label(cblab, fontsize=14)

            # Add vertical lines at the position where this track crossed the climatological front positions.
            SAFcfrontidx = near2(lonSAF, latSAF, SAFlonp, SAFlatp, return_index=True)[0]
            PFcfrontidx = near2(lonPF, latPF, PFlonp, PFlatp, return_index=True)[0]
            SACCFcfrontidx = near2(lonSACCF, latSACCF, SACCFlonp, SACCFlatp, return_index=True)[0]

            ax1.axvline(x=xSAF[SAFcfrontidx], color="gray", linestyle="dotted")
            ax2.axvline(x=xPF[PFcfrontidx], color="gray", linestyle="dotted")
            ax3.axvline(x=xSACCF[SACCFcfrontidx], color="gray", linestyle="dotted")

            ax1.text(0.01, 0.01, "SAF", fontsize=15, transform=ax1.transAxes)
            ax2.text(0.01, 0.01, "PF", fontsize=15, transform=ax2.transAxes)
            ax3.text(0.01, 0.01, "SACCF", fontsize=15, transform=ax3.transAxes)
            ax3.set_xlabel("Along-track distance [km]", fontsize=15)
            ax2.set_ylabel("Depth [m]", fontsize=15)
            ax1.set_title("%s, %s: %s $\longrightarrow$ %s"%(fn, iostr_short, tsstr, testr), y=1, fontsize=9, x=0.635)
            fig.savefig("ACCjets/contour/%s_%s.png"%(fn, iostr_short), bbox_inches="tight", dpi=175)
            plt.close(fig)

        # Plot cruise track within the Drake Passage lat-lon box.
        if PLT_MAP:
            figm, axm = plt.subplots(subplot_kw=dict(projection=proj))
            if ROTATE_UV_USING_ADT:
                timei_adt = Timestamp(time[time.size//2])
                adti = adt.interp(dict(time=timei_adt), method="linear")
                adtmax = np.abs(adti).max().values
                axm.pcolormesh(adti["longitude"], adti["latitude"], adti.values, cmap=balance, zorder=0, vmin=-adtmax, vmax=adtmax)
                adti.plot.contour(ax=axm, levels=70, linestyles="solid", linewidths=0.2, colors="k", zorder=1, vmin=-adtmax, vmax=adtmax)
            axm.plot(loncSAF_plt, latcSAF_plt, "gray", zorder=2)
            axm.plot(loncPF_plt, latcPF_plt, "gray", zorder=2)
            axm.plot(loncSACCF_plt, latcSACCF_plt, "gray", zorder=2)
            axm.plot(lon, lat, "k", zorder=3)
            if DO_SAF:
                axm.plot(SAFxing.x, SAFxing.y, linestyle="none", marker="+", ms=7, mfc="r", mec="r", zorder=3)
                axm.plot(loncoreSAF, latcoreSAF, linestyle="none", marker="x", ms=7, mfc="r", mec="r", zorder=4)
            if DO_PF:
                axm.plot(PFxing.x, PFxing.y, linestyle="none", marker="+", ms=7, mfc="g", mec="g", zorder=4)
                axm.plot(loncorePF, latcorePF, linestyle="none", marker="x", ms=7, mfc="g", mec="g", zorder=4)
            if DO_SACCF:
                axm.plot(SACCFxing.x, SACCFxing.y, linestyle="none", marker="+", ms=7, mfc="b", mec="b", zorder=4)
                axm.plot(loncoreSACCF, latcoreSACCF, linestyle="none", marker="x", ms=7, mfc="b", mec="b", zorder=4)
            axm.coastlines(zorder=9)
            axm.set_title("%s, %s $\longrightarrow$ %s"%(iostr, tsstr, testr))
            figm.savefig("ACCjets/inout_tracks/%s_%s.png"%(fn, iostr_short), bbox_inches="tight", dpi=250)
            plt.close(figm)

        n += 1

# Interpolate Ld from LaCasce & Groeskamp (2020) along (lon, lat) sections of each occupation.
dd = loadmat("../data/misc/Ld_LaCasce-Groeskamp_2020.mat")
lonLd, latLd, Ldflati, Ldsurfi = dd["xt"].squeeze(), dd["yt"].squeeze(), dd["Ld_flat_RK4"].squeeze().T*1e-3, dd["Ld_rough_RK4"].squeeze().T*1e-3
xiLd, yiLd = np.meshgrid(lonLd, latLd)

fg = np.isfinite(Ldflati)
mskflati = ~fg
pts_flat, z_flat = (xiLd[fg], yiLd[fg]), Ldflati[fg]
fg = np.isfinite(Ldsurfi)
msksurfi = ~fg
pts_surf, z_surf = (xiLd[fg], yiLd[fg]), Ldsurfi[fg]

LdflatsSAF = lonsSAF.copy()*np.nan
LdflatsPF = lonsPF.copy()*np.nan
LdflatsSACCF = lonsSACCF.copy()*np.nan
LdsurfsSAF = lonsSAF.copy()*np.nan
LdsurfsPF = lonsPF.copy()*np.nan
LdsurfsSACCF = lonsSACCF.copy()*np.nan
ntSAF, ntPF, ntSACCF = tcsSAF.size, tcsPF.size, tcsSACCF.size

print("")
for n in range(ntSAF):
    print("Interpolating SAF Ld %d / %d"%(n+1, ntSAF))
    ipts = (lon180to360(lonsSAF[n, :]), latsSAF[n, :])
    LdflatsSAF[n, :] = griddata(pts_flat, z_flat, ipts, method="cubic")
    LdsurfsSAF[n, :] = griddata(pts_surf, z_surf, ipts, method="cubic")

print("")
for n in range(ntPF):
    print("Interpolating PF Ld %d / %d"%(n+1, ntPF))
    ipts = (lon180to360(lonsPF[n, :]), latsPF[n, :])
    LdflatsPF[n, :] = griddata(pts_flat, z_flat, ipts, method="cubic")
    LdsurfsPF[n, :] = griddata(pts_surf, z_surf, ipts, method="cubic")

print("")
for n in range(ntSACCF):
    print("Interpolating SACCF Ld %d / %d"%(n+1, ntSACCF))
    ipts = (lon180to360(lonsSACCF[n, :]), latsSACCF[n, :])
    LdflatsSACCF[n, :] = griddata(pts_flat, z_flat, ipts, method="cubic")
    LdsurfsSACCF[n, :] = griddata(pts_surf, z_surf, ipts, method="cubic")

# Save to netCDF files.
tcsSAFc = IndexVariable("t", tcsSAF, attrs=dict(long_name="Time"))
tcsPFc = IndexVariable("t", tcsPF, attrs=dict(long_name="Time"))
tcsSACCFc = IndexVariable("t", tcsSACCF, attrs=dict(long_name="Time"))
xsc = IndexVariable("x", xs, attrs=dict(units="km", long_name="Cross-stream distance"))

coords_SAF = dict(t=tcsSAFc, x=xsc)
coords_PF = dict(t=tcsPFc, x=xsc)
coords_SACCF = dict(t=tcsSACCFc, x=xsc)
coordsx = dict(x=xsc)
coordst_SAF = dict(t=tcsSAFc)
coordst_PF = dict(t=tcsPFc)
coordst_SACCF = dict(t=tcsSACCFc)

U_SAF = DataArray(usSAF, coords=coords_SAF, attrs=dict(units="m/s", long_name="Downstream velocity"))
U_PF = DataArray(usPF, coords=coords_PF, attrs=dict(units="m/s", long_name="Downstream velocity"))
U_SACCF = DataArray(usSACCF, coords=coords_SACCF, attrs=dict(units="m/s", long_name="Downstream velocity"))
Lon_SAF = DataArray(lonsSAF, coords=coords_SAF, attrs=dict(units="Degrees east", long_name="Cross-stream longitudes"))
Lat_SAF = DataArray(latsSAF, coords=coords_SAF, attrs=dict(units="Degrees north", long_name="Cross-stream latitudes"))
Lon_PF = DataArray(lonsPF, coords=coords_PF, attrs=dict(units="Degrees east", long_name="Cross-stream longitudes"))
Lat_PF = DataArray(latsPF, coords=coords_PF, attrs=dict(units="Degrees north", long_name="Cross-stream latitudes"))
Lon_SACCF = DataArray(lonsSACCF, coords=coords_SACCF, attrs=dict(units="Degrees east", long_name="Cross-stream longitudes"))
Lat_SACCF = DataArray(latsSACCF, coords=coords_SACCF, attrs=dict(units="Degrees north", long_name="Cross-stream latitudes"))
Ldflat_SAF = DataArray(LdflatsSAF, coords=coords_SAF, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf_SAF = DataArray(LdsurfsSAF, coords=coords_SAF, attrs=dict(units="km", long_name="First surface deformation radius"))
Ldflat_PF = DataArray(LdflatsPF, coords=coords_PF, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf_PF = DataArray(LdsurfsPF, coords=coords_PF, attrs=dict(units="km", long_name="First surface deformation radius"))
Ldflat_SACCF = DataArray(LdflatsSACCF, coords=coords_SACCF, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf_SACCF = DataArray(LdsurfsSACCF, coords=coords_SACCF, attrs=dict(units="km", long_name="First surface deformation radius"))

dvars_SAF = dict(us=U_SAF, lon=Lon_SAF, lat=Lat_SAF, Ldflat=Ldflat_SAF, Ldsurf=Ldsurf_SAF)
dvars_PF = dict(us=U_PF, lon=Lon_PF, lat=Lat_PF, Ldflat=Ldflat_PF, Ldsurf=Ldsurf_PF)
dvars_SACCF = dict(us=U_SACCF, lon=Lon_SACCF, lat=Lat_SACCF, Ldflat=Ldflat_SACCF, Ldsurf=Ldsurf_SACCF)
if ROTATE_UV_USING_ADT:
    Angdiff_SAF = DataArray(angdiffsSAF, coords=coordst_SAF, attrs=dict(units="Degrees", long_name="Difference between surface geostrophic velocity vector from ADT and direction normal to the ship track"))
    Angdiff_PF = DataArray(angdiffsPF, coords=coordst_PF, attrs=dict(units="Degrees", long_name="Difference between surface geostrophic velocity vector from ADT and direction normal to the ship track"))
    Angdiff_SACCF = DataArray(angdiffsSACCF, coords=coordst_SACCF, attrs=dict(units="Degrees", long_name="Difference between surface geostrophic velocity vector from ADT and direction normal to the ship track"))
    dvars_SAF.update(dict(angdiff=Angdiff_SAF))
    dvars_PF.update(dict(angdiff=Angdiff_PF))
    dvars_SACCF.update(dict(angdiff=Angdiff_SACCF))

dsSAF = Dataset(data_vars=dvars_SAF).sortby("t")
dsSAF.to_netcdf("../data/derived/ustream_SAFjet_LMG.nc")
dsPF = Dataset(data_vars=dvars_PF).sortby("t")
dsPF.to_netcdf("../data/derived/ustream_PFjet_LMG.nc")
dsSACCF = Dataset(data_vars=dvars_SACCF).sortby("t")
dsSACCF.to_netcdf("../data/derived/ustream_SACCFjet_LMG.nc")
