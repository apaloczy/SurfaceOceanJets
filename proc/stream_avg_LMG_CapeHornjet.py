# Calculate stream-averaged velocity for shelfbreak jet south of Cape Horn
# observed in shipboard ADCP data (Laurence M. Gould).
import numpy as np
import matplotlib.pyplot as plt
from xarray import open_dataset, DataArray, Dataset, IndexVariable
from pandas import Timestamp, Timedelta
from hdf5storage import loadmat
from gsw import distance
from cmocean.cm import balance, thermal
from glob import glob
from scipy.interpolate import griddata, interp1d
from ap_tools.utils import lon360to180, lon180to360, near, compass2trig, rot_vec
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import LAND
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from gsw import f as fcor
import warnings
warnings.filterwarnings("ignore")


def rolltrim2(x, n):
    if n!=0:
        x = np.roll(x, n, axis=1)
        if n>0:
            x[:, :n] = np.nan
        else:
            x[:, n:] = np.nan
    else:
        pass

    return x


def get_isoT_depth(z, T, T0):
    N = T.shape[1]
    if not np.all(np.diff(z)>0):
        z = np.flipud(z)
        T = np.flipud(T)

    zT0 = np.empty(N)*np.nan
    for i in range(N):
        Ti = T[:, i]
        fgi = np.isfinite(Ti)
        if fgi.any():
            Timin, Timax = np.nanmin(Ti), np.nanmax(Ti)
            if np.logical_and(T0>=Timin, T0<=Timax):
                zT0[i] = np.interp(T0, Ti[fgi], z[fgi], left=np.nan, right=np.nan)

    return zT0


def interp_secz(xT, zT, T, ziT):
    flipped = False
    if not np.all(np.diff(zT)>0):
        zT = np.flipud(zT)
        T = np.flipud(T)
        flipped = True

    if not np.all(np.diff(ziT)>0):
        ziT = np.flipud(ziT)

    nziT = ziT.size
    nxT = xT.size
    Ti = np.empty((nziT, nxT))*np.nan
    for i in range(nxT):
        fgi = np.isfinite(T[:, i])
        if fgi.any():
            Ti[:, i] = np.interp(ziT, zT[fgi], T[fgi, i], left=np.nan, right=np.nan)

    if flipped:
        Ti = np.flipud(Ti)

    return Ti


def interp_sec(xT, zT, T, xiT, xADCP, fc, msktopo=True):
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

    fci = near(xiT, xADCP[fc], return_index=True)
    nshft = nxiTh - fci
    Tiaux = rolltrim2(Ti, nshft)

    return Tiaux


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
        fig.savefig("CapeHornjet/Ld_map_%s.png"%name, bbox_inches="tight", dpi=200)


#---
plt.close("all")

head = "../data/ADCP_LMG/"
maxADCPzbins = 50
nXBTs_max = 15
dtADCP_thresh = 0.5 # [days]
SAVEFIGS = True
PLT_MAP = True
shelfx = 20 # [km]
offshx = 20 # [km]
Llr = 10    # [km]

nobsTi_thresh = 10
dTcc2 = 0.5

# Zoom in on just the straight-ish part of the cross-shelfbreak section.
xmin, xmax = -65.1, -64.8
ymin, ymax = -55.29, -54.63

ntopbins_avg = 4 # Top 50 m.
umax = 1.5
angcone = 25
minpts = 4
Ls = 100 # [km]
dxs = 1.0 # [km]

xs = np.linspace(-Ls/2, 0, num=(round(Ls/dxs/2)+1))
xs = np.append(xs, -np.flipud(xs[:-1]))

sconel, sconer = 180 + angcone, 180 - angcone
nconel, nconer = 360 - angcone, angcone

# read list of bad transects to skip over when doing the stream averaging.

f = open("../data/misc/bad_cruises_shelfbreak.txt", mode="r")
aux = f.readlines()
bad_cruises = [l.strip("\n") for l in aux]
f.close()

f = open("../data/misc/leftshift_cruises_shelfbreak.txt", mode="r")
aux = f.readlines()
leftshift_cruises = [l.strip("\n") for l in aux]
f.close()

proj = ccrs.PlateCarree()

if PLT_MAP:
    figm, axm = plt.subplots(subplot_kw=dict(projection=proj))

head_XBT = "../data/XBT_LMG/"
fnames_XBT = glob(head_XBT + "*.nc")
fnames_XBT.sort()

TccXBT = np.arange(-2, 10)
tsXBT, teXBT = [], []
for f in fnames_XBT:
    tiXBT = open_dataset(f)["t"].values
    tsXBT.append(tiXBT[0])
    teXBT.append(tiXBT[-1])
tsXBT, teXBT = map(np.array, (tsXBT, teXBT))

fnames = glob(head + "*_short.nc")
fnames.sort()
nf = len(fnames)
badTflag = -0.999

dxiT = 1.0 # [km]
xLT = 150  # [km]
xiT = np.arange(0, xLT + dxiT, dxiT)
xiT = np.concatenate((-np.flipud(xiT[1:]), xiT))
nxiT = xiT.size
nxiTh = nxiT//2

all_dxs = np.array([])
lons_out, lats_out, fcores_out = np.array([]), np.array([]), np.array([])
lons_in, lats_in, fcores_in = np.array([]), np.array([]), np.array([])
us = None
Tiauxs = None
for f in fnames:
    fn = f.split("/")[-1][:5]
    ds = open_dataset(f).set_coords(("lat", "lon"))
    lon, lat, time = ds["lon"].values, ds["lat"].values, ds["time"].values
    hdng = ds["heading"].values
    ts, te = str(Timestamp(time.min())).split(" ")[0], str(Timestamp(time.max())).split(" ")[0]
    z = -ds["depth"].values
    u, v = ds["u"].values, ds["v"].values

    fbb = np.logical_and(np.logical_and(lon>=xmin, lon<=xmax), np.logical_and(lat>=ymin, lat<=ymax))
    lon, lat, time, hdng = lon[fbb], lat[fbb], time[fbb], hdng[fbb]
    z, u, v = z[fbb, :], u[fbb, :], v[fbb, :]

    # Plot heading within the shelfbreak lat-lon box
    # to separating outbound and inbound parts of the cruise.
    hdng = lon180to360(hdng)

    fg = np.logical_and(np.isfinite(lon), np.isfinite(lat))
    lon, lat, time, z = lon[fg], lat[fg], time[fg], z[fg, :]
    u, v = u[fg, :], v[fg, :]

    # Transects as a function of along-track distance.
    fout = np.logical_and(hdng>=sconer, hdng<=sconel)
    fin = np.logical_or(hdng>=nconel, hdng<=nconer)

    # dt Should be around 5 min between ADCP profiles. Flag points much farther apart than that.
    dtqcout = np.abs(np.array([Timedelta(dti).total_seconds()/86400 for dti in np.diff(time[fout])]))
    dtqcin = np.abs(np.array([Timedelta(dti).total_seconds()/86400 for dti in np.diff(time[fin])]))
    fgout = dtqcout>dtADCP_thresh
    fgin = dtqcin>dtADCP_thresh
    CAPOUT, CAPIN = False, False
    if fgout.any():
        print("Removed %d time-isolated outbound points."%fgout.sum())
        fgout = np.where(~fgout)[0]
        CAPOUT = True
    if fgin.any():
        print("Removed %d time-isolated inbound points."%fgin.sum())
        fgin = np.where(np.flipud(~fgin))[0]
        CAPIN = True

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
    axl, axr = ax

    # Outbound leg.
    if fout.sum()>minpts and fn + "-O" not in bad_cruises:
        lonout, latout, timeout = lon[fout], lat[fout], time[fout]
        hdngout = hdng[fout]
        if CAPOUT:
            lonout, latout, timeout, hdngout = lonout[fgout], latout[fgout], timeout[fgout], hdngout[fgout]
        xout = np.append(0, np.nancumsum(distance(lonout, latout)))*1e-3
        zout = z[fout, :]
        uout, vout = u[fout, :], v[fout, :]
        if CAPOUT:
            zout, uout, vout = zout[fgout, :], uout[fgout, :], vout[fgout, :]
        xpout = np.tile(xout[:, np.newaxis], (1, zout.shape[1]))
        uoutu = uout
        cblab = "$u$ east [m/s]"

        # Locate shelfbreak jet core.
        uoutusurf_aux = np.nanmean(uoutu[:, :ntopbins_avg], axis=1) # Average top few bins.
        fshelf = xout<=shelfx # Excluding shelf, there are often zonal velocity maxima there too.
        if fn + "-O" != "00787-O": # This transect starts closer to the shelfbreak.
            uoutusurf_aux[fshelf] = 0

        if fn + "-O" in leftshift_cruises:
            foffsh = np.abs(xout - xout[-1])<=offshx # Excluding deep part of transect, there are often zonal velocity maxima there too.
            uoutusurf_aux[foffsh] = 0

        fcore = np.nanargmax(uoutusurf_aux)
        xstream = xout - xout[fcore]
        zout0 = zout[0, :]
        uaux = interp_sec(xout, zout0, uout.T, xiT, xout, fcore)
        if zout0.size>maxADCPzbins:
            print("%d is more than the max %d ADCP bins. Capping."%(zout0.size, maxADCPzbins))
            uaux = uaux[:maxADCPzbins, :]

        lonouti = np.interp(xiT, xstream, lonout, left=np.nan, right=np.nan)
        latouti = np.interp(xiT, xstream, latout, left=np.nan, right=np.nan)
        timei = timeout[timeout.size//2]
        if us is None:
            us = uaux[..., np.newaxis]
            lons = lonouti[np.newaxis, :]
            lats = latouti[np.newaxis, :]
            ti = timei
        else:
            us = np.dstack((us, uaux[..., np.newaxis]))
            lons = np.vstack((lons, lonouti[np.newaxis, :]))
            lats = np.vstack((lats, latouti[np.newaxis, :]))
            ti = np.append(ti, timei)

        # Get collocated XBT T transect.
        fXBT_transect = None
        for tsT, teT in zip(tsXBT, teXBT):
            fXBT = np.logical_and(timeout>=tsT, timeout<=teT)
            if fXBT.any():
                fXBT_transect = np.where(tsXBT==tsT)[0][0]

        if fXBT_transect is not None:
            dsXBT = open_dataset(fnames_XBT[fXBT_transect])
            tT = dsXBT["t"]
            fT = np.logical_and(tT>=timeout[0], tT<=timeout[-1]).values
            nXBTs = fT.sum().flatten()[0]
            if nXBTs>nXBTs_max:
                print("Too many XBT profiles (%d). Skipping."%nXBTs)
                continue
            if nXBTs>1:
                print("Outbound %s -> XBT profiles in shelfbreak jet: "%fn, nXBTs)
                lonT, latT = dsXBT["lon"].values[fT], dsXBT["lat"].values[fT]
                xT = np.append(0, np.cumsum(distance(lonT, latT)))*1e-3
                fXBT_closest = near(latout, latT[0], return_index=True)
                xADCP0 = xout[fXBT_closest]
                xT = xT + xADCP0
                zT = dsXBT["z"].values
                T = dsXBT["T"].values[:, fT]
                T[T==badTflag] = np.nan
                cc = axl.contour(xT, zT, T, levels=TccXBT, colors="gray", zorder=999)
                axl.clabel(cc)
                _ = [axl.axvline(x=xTi, color="gray", alpha=0.2) for xTi in xT]

                # Interpolate T in stream coordinates to calculate time-averaged T section.
                Tiaux = interp_sec(xT, zT, T, xiT, xout, fcore)
                if Tiauxs is None:
                    Tiauxs = Tiaux[..., np.newaxis]
                else:
                    Tiauxs = np.dstack((Tiauxs, Tiaux[..., np.newaxis]))

        axl.set_xlim(0, xout[-1])
        cs = axl.pcolormesh(xpout, zout, uoutu, vmin=-umax, vmax=umax, cmap=balance)
        axl.axvline(x=xout[fcore], color="gray", linestyle="dashed")
        cax = axl.inset_axes([0.10, 0.01, 0.25, 0.02])
        cbl = fig.colorbar(cs, cax=cax, orientation="horizontal", ticklocation="top")
        cbl.set_label(cblab, fontsize=14)
        axl.set_xlabel("Along-track distance [km]", fontsize=15, x=1)
        axl.set_ylabel("Depth [m]", fontsize=15)
        axl.set_title("Outbound", fontsize=15)
        cm = inset_axes(axl, width="60%", height="40%", loc="center left", axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
        cm.plot(lon, lat, color="gray", linewidth=0.5, marker=".", ms=1)
        cm.plot(lonout, latout, color="r", linewidth=0.5, marker=".", ms=1)

        # Plot cruise track within the shelfbreak lat-lon box.
        if PLT_MAP:
            axm.plot(lonout, latout, linestyle="none", marker=".", ms=1, mfc="b", mec="b")

        lons_out = np.append(lons_out, lonout[fcore])
        lats_out = np.append(lats_out, latout[fcore])
        fcores_out = np.append(fcores_out, fcore)
        all_dxs = np.append(all_dxs, np.diff(xout))

    # Inbound leg.
    if fin.sum()>minpts and fn + "-I" not in bad_cruises:
        lonin, latin, timein = np.flipud(lon[fin]), np.flipud(lat[fin]), np.flipud(time[fin])
        hdngin = np.flipud(hdng[fin])
        zin = np.flipud(z[fin, :])
        uin, vin = np.flipud(u[fin, :]), np.flipud(v[fin, :])
        if CAPIN:
            lonin, latin, timein, hdngin = lonin[fgin], latin[fgin], timein[fgin], hdngin[fgin]
            zin, uin, vin = zin[fgin, :], uin[fgin, :], vin[fgin, :]
        xin = np.append(0, np.nancumsum(distance(lonin, latin)))*1e-3
        xpin = np.tile(xin[:, np.newaxis], (1, zin.shape[1]))
        uinu = uin
        cblab = "$u$ east [m/s]"

        # Locate shelfbreak jet core.
        uinusurf_aux = np.nanmean(uinu[:, :ntopbins_avg], axis=1) # Average top 2 bins.
        fshelf = xin<=shelfx # Excluding shelf, there are often zonal velocity maxima there too.
        foffsh = np.abs(xin - xin[-1])<=offshx # Excluding deep part of transect, there are often zonal velocity maxima there too.
        uinusurf_aux[fshelf] = 0

        if fn + "-I" in leftshift_cruises:
            foffsh = np.abs(xin - xin[-1])<=offshx # Excluding deep part of transect, there are often zonal velocity maxima there too.
            uinusurf_aux[foffsh] = 0

        fcore = np.nanargmax(uinusurf_aux)
        xstream = xin - xin[fcore]
        zin0 = zin[0, :]
        uaux = interp_sec(xin, zin0, uin.T, xiT, xin, fcore)
        if zin0.size>maxADCPzbins:
            print("%d is more than the max %d ADCP bins. Capping."%(zin0.size, maxADCPzbins))
            uaux = uaux[:maxADCPzbins, :]

        lonini = np.interp(xiT, xstream, lonin, left=np.nan, right=np.nan)
        latini = np.interp(xiT, xstream, latin, left=np.nan, right=np.nan)
        timei = timein[timein.size//2]
        if us is None:
            us = uaux[..., np.newaxis]
            lons = lonini[np.newaxis, :]
            lats = latini[np.newaxis, :]
            ti = timei
        else:
            us = np.dstack((us, uaux[..., np.newaxis]))
            lons = np.vstack((lons, lonini[np.newaxis, :]))
            lats = np.vstack((lats, latini[np.newaxis, :]))
            ti = np.append(ti, timei)

        # Get collocated XBT T transect.
        fXBT_transect = None
        for tsT, teT in zip(tsXBT, teXBT):
            fXBT = np.logical_and(timein>=tsT, timein<=teT)
            if fXBT.any():
                fXBT_transect = np.where(tsXBT==tsT)[0][0]

        if fXBT_transect is not None:
            dsXBT = open_dataset(fnames_XBT[fXBT_transect])
            tT = dsXBT["t"]
            fT = np.logical_and(tT>=timein[-1], tT<=timein[0]).values
            nXBTs = fT.sum().flatten()[0]
            if nXBTs>nXBTs_max:
                print("Too many XBT profiles (%d). Skipping."%nXBTs)
                continue
            if nXBTs>1:
                print("Inbound %s  <- XBT profiles in shelfbreak jet: "%fn, nXBTs)
                lonT, latT = dsXBT["lon"].values[fT], dsXBT["lat"].values[fT]
                lonT, latT = map(np.flipud, (lonT, latT))
                xT = np.append(0, np.cumsum(distance(lonT, latT)))*1e-3
                fXBT_closest = near(latin, latT[0], return_index=True)
                xADCP0 = xin[fXBT_closest]
                xT = xT + xADCP0
                zT = dsXBT["z"].values
                T = dsXBT["T"].values[:, fT]
                T[T==badTflag] = np.nan
                T = np.fliplr(T)
                cc = axr.contour(xT, zT, T, levels=TccXBT, colors="gray", zorder=999)
                axr.clabel(cc)
                _ = [axr.axvline(x=xTi, color="gray", alpha=0.2) for xTi in xT]

                # Interpolate T in stream coordinates to calculate time-averaged T section.
                Tiaux = interp_sec(xT, zT, T, xiT, xin, fcore)
                if Tiauxs is None:
                    Tiauxs = Tiaux[..., np.newaxis]
                else:
                    Tiauxs = np.dstack((Tiauxs, Tiaux[..., np.newaxis]))

        axr.set_xlim(0, xin[-1])
        cs = axr.pcolormesh(xpin, zin, uinu, vmin=-umax, vmax=umax, cmap=balance)
        axr.axvline(x=xin[fcore], color="gray", linestyle="dashed")
        cax = axr.inset_axes([0.10, 0.01, 0.25, 0.02])
        cbr = fig.colorbar(cs, cax=cax, orientation="horizontal", ticklocation="top")
        cbr.set_label(cblab, fontsize=14)
        axr.set_title("Inbound", fontsize=15)
        cm = inset_axes(axr, width="60%", height="40%", loc="center left", axes_class=GeoAxes, axes_kwargs=dict(map_projection=proj))
        cm.plot(lon, lat, color="gray", linewidth=0.5, marker=".", ms=1)
        cm.plot(lonin, latin, color="r", linewidth=0.5, marker=".", ms=1)

        # Plot cruise track within the shelfbreak lat-lon box.
        if PLT_MAP:
            axm.plot(lonin, latin, linestyle="none", marker=".", ms=1, mfc="r", mec="r")

        lons_in = np.append(lons_in, lonin[fcore])
        lats_in = np.append(lats_in, latin[fcore])
        fcores_in = np.append(fcores_in, fcore)
        all_dxs = np.append(all_dxs, np.diff(xin))

    fig.subplots_adjust(wspace=0.025)
    fig.suptitle("Cruise %s:     %s $\longrightarrow$ %s"%(fn, ts, te), y=1, fontsize=15)
    if SAVEFIGS:
        fig.savefig("CapeHornjet/figs_outbound_inbound/%s.png"%fn, bbox_inches="tight")
        plt.close()

all_lons = np.append(lons_in, lons_out)
all_lats = np.append(lats_in, lats_out)

if PLT_MAP:
    dl = 0.3
    axm.set_extent((xmin-dl, xmax+dl, ymin-dl, ymax+dl))
    axm.coastlines()
    axm.set_title("All cruise tracks", fontsize=16, fontweight="black")
    if SAVEFIGS:
        figm.savefig("CapeHornjet/cruise_tracks_shelfbreak.png", bbox_inches="tight", dpi=150)

# Add left and right surface Ld from LG20 to stream average plot****
dd = loadmat("../data/misc/Ld_LaCasce-Groeskamp_2020.mat")
lonLd, latLd, Ldflati, Ldsurfi = dd["xt"].squeeze(), dd["yt"].squeeze(), dd["Ld_flat_RK4"].squeeze().T*1e-3, dd["Ld_rough_RK4"].squeeze().T*1e-3
xiLd, yiLd = np.meshgrid(lonLd, latLd)

xsiT = xiT - xiT[nxiTh]

nlr = round(Llr/dxs)
dlat = Llr/111.12 # [km/deg latitude]
nxh = xsiT.size//2
lonc180 = all_lons.mean()
lonc, latc = lon180to360(lonc180)[0], all_lats.mean()
lonl, lonr = lonc, lonc
latl, latr = latc + dlat, latc - dlat

fg = np.isfinite(Ldflati)
mskflati = ~fg
Ldflatl, Ldflatr = griddata((xiLd[fg], yiLd[fg]), Ldflati[fg], ([lonl, lonr], [latl, latr]), method="cubic")
fg = np.isfinite(Ldsurfi)
msksurfi = ~fg
Ldsurfl, Ldsurfr = griddata((xiLd[fg], yiLd[fg]), Ldsurfi[fg], ([lonl, lonr], [latl, latr]), method="cubic")
Ldflati[mskflati] = np.nan
Ldsurfi[msksurfi] = np.nan

# Get bathymetry.
fbathymetry="../data/srtm15p/SRTM15_V2.5.5.nc"
isobs = (200, 2000, 4000)
d = 0.5
xmi, xma = lonc180 - d, lonc180 + d
ymi, yma = latc - d, latc + d
ds = open_dataset(fbathymetry).sel(lon=slice(xmi, xma), lat=slice(ymi, yma))
xt, yt = np.meshgrid(ds["lon"].values, ds["lat"].values)
zt = -ds["z"].values

pltmaps_Ld(lonc, latc, lonl, latl, lonr, latr, "shelfbreak_jet", Ldflatl, Ldflatr, Ldsurfl, Ldsurfr, dxy=d, SAVEFIG=True)

# Number of effective degrees of freedom: Assuming every other cruise as an independent realization.
xEDOF = np.isfinite(us).sum(axis=2)

usavg = np.nanmean(us, axis=2)
SE = np.nanstd(us, axis=2)/np.sqrt(xEDOF)
CL95l = usavg - 2*SE
CL95u = usavg + 2*SE

# Plot full averaged velocity section.
umaxpl = 1.0
du = 0.2
ucc = np.arange(-umaxpl, umaxpl+du, du)
z0 = z[0, :]
ztop, zbot = -z[0, 0], -z[0, ntopbins_avg - 1]
f_not_significant = CL95l*CL95u<0 # Points where the time-averaged velocity is not statistically different from zero.
fz, fx = np.where(f_not_significant)
usavg_plt = usavg.copy()
usavg_plt[fz, fx] = np.nan

fig, ax = plt.subplots()
ax.pcolormesh(xsiT, z0, usavg_plt, vmin=-umaxpl, vmax=umaxpl, cmap=balance)
cc = ax.contour(xsiT, z0, usavg_plt, levels=ucc, colors="gray")
# ax.plot(xsiT[fx], z0[fz], linestyle="none", marker="x", mfc="gray", mec="gray", ms=4)
ax.clabel(cc)
ax.set_xlim(xsiT[0], xsiT[-1])
ax.axhline(y=-ztop, color="gray", linestyle="solid", linewidth=0.5)
ax.axhline(y=-zbot, color="gray", linestyle="solid", linewidth=0.5)
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title("Shelfbreak jet (1999-2018 XBT average)", fontsize=12)
fig.savefig("CapeHornjet/shelfbreakjet_umeansec.png", bbox_inches="tight", dpi=150)

z = z.T # Depths of ADCP bins.
fzu = np.logical_and(z0>=-zbot, z0<=-ztop)
fzT = np.logical_and(zT>=-zbot, zT<=-ztop) # Depths of XBT levels within the ADCP bin depth range.
CL95lsurf = np.nanmean(CL95l[fzu, :], axis=0)
CL95usurf = np.nanmean(CL95u[fzu, :], axis=0)
usavgsurf = np.nanmean(usavg[fzu, :], axis=0)

xsclip = 60 # [km]
xs = xsiT.copy()
usavgsurf[np.abs(xs)>xsclip] = np.nan
CL95lsurf[np.abs(xs)>xsclip] = np.nan
CL95usurf[np.abs(xs)>xsclip] = np.nan

tic = IndexVariable("t", ti, attrs=dict(long_name="Time"))
xsc = IndexVariable("x", xs, attrs=dict(units="km", long_name="Cross-stream distance"))
zTc = IndexVariable("z", zT, attrs=dict(units="m", long_name="Depth"))
coords = dict(t=tic, x=xsc)
coordsT = dict(z=zTc, x=xsc)
coordsx = dict(x=xsc)
us = us.swapaxes(0, 1)
uszavg = np.nanmean(us[:, fzu, :], axis=1).T

U = DataArray(uszavg, coords=coords, attrs=dict(units="m/s", long_name="Downstream velocity"))
Lon = DataArray(lons, coords=coords, attrs=dict(units="Degrees east", long_name="Cross-stream longitudes"))
Lat = DataArray(lats, coords=coords, attrs=dict(units="Degrees north", long_name="Cross-stream latitudes"))

fg = np.isfinite(Ldflati)
mskflati = ~fg
pts_flat, z_flat = (xiLd[fg], yiLd[fg]), Ldflati[fg]
fg = np.isfinite(Ldsurfi)
msksurfi = ~fg
pts_surf, z_surf = (xiLd[fg], yiLd[fg]), Ldsurfi[fg]

Ldflats = lons.copy()*np.nan
Ldsurfs = lons.copy()*np.nan
nt = ti.size

print("")
for n in range(nt):
    print("Interpolating  Ld %d / %d"%(n+1, nt))
    ipts = (lon180to360(lons[n, :]), lats[n, :])
    Ldflats[n, :] = griddata(pts_flat, z_flat, ipts, method="cubic")
    Ldsurfs[n, :] = griddata(pts_surf, z_surf, ipts, method="cubic")

Ldflat = DataArray(Ldflats, coords=coords, attrs=dict(units="km", long_name="First baroclinic deformation radius"))
Ldsurf = DataArray(Ldsurfs, coords=coords, attrs=dict(units="km", long_name="First surface deformation radius"))

xEDOF = np.isfinite(us).sum(axis=2)
xEDOF = xEDOF.swapaxes(0, 1)
xEDOF_aux = np.float32(xEDOF).copy()
xEDOF_aux[:, np.abs(xs)>xsclip] = np.nan
dyl, dyt = 0.1, 0.1
fig, ax = plt.subplots()
ax.axhline(y=0, color="gray", linestyle="dashed")
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.fill_between(xsiT, CL95lsurf, CL95usurf, color="k", alpha=0.2)
ax.plot(xsiT, usavgsurf, "k")
ax.set_xlim(-xsclip, xsclip)
ax.set_ylim(np.nanmin(usavgsurf) - dyl, np.nanmax(usavgsurf) + dyl)
ax.text(0.01, 0.95, "Max/min eDOF = %d / %d"%(np.nanmax(xEDOF_aux), np.nanmin(xEDOF_aux)), fontsize=10, transform=ax.transAxes)
ax.text(0.2, 0.75+dyt, "$L_{dl}$ = %.1f, %.1f km"%(Ldsurfl, Ldflatl), fontsize=12, transform=ax.transAxes, ha="center", color="r")
ax.text(0.8, 0.75+dyt, "$L_{dr}$ = %.1f, %.1f km"%(Ldsurfr, Ldflatr), fontsize=12, transform=ax.transAxes, ha="center", color="g")
ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
ax.set_ylabel("Eastward velocity $u$ [m/s]", fontsize=15)
ax.set_title("Shelfbreak jet (1999-2018 underway ADCP average, %d-%d m)"%(ztop, zbot), fontsize=10)
fig.savefig("CapeHornjet/shelfbreakjet_umean.png", bbox_inches="tight", dpi=125)

ddxusavg = np.gradient(usavgsurf)/(np.gradient(xsiT)*1e3)
f0 = fcor(latc)

fig, ax = plt.subplots()
ax.axhline(y=0, color="gray", linestyle="dashed")
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.plot(xs, ddxusavg/f0, "k")
ax.set_xlim(-xsclip, xsclip)
ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
ax.set_ylabel("Normalized vorticity $\zeta/f$ [unitless]", fontsize=15)
ax.set_title("Shelfbreak jet (1999-2018 underway ADCP average, %d-%d m)"%(ztop, zbot), fontsize=10)
fig.savefig("CapeHornjet/shelfbreakjet_zetamean.png", bbox_inches="tight", dpi=125)

# Time-averaged XBT temperature section.
TccXBT2 = np.arange(TccXBT[0], TccXBT[-1] + dTcc2, dTcc2)
nobsTi = np.isfinite(Tiauxs).sum(axis=2)
Tsmean = np.nanmean(Tiauxs, axis=2)
Tsmean[nobsTi<nobsTi_thresh] = np.nan

fig, ax = plt.subplots()
ax.pcolormesh(xsiT, zT, Tsmean, cmap=thermal)
cc = ax.contour(xsiT, zT, Tsmean, levels=TccXBT2, colors="gray")
ax.clabel(cc)
ax.set_xlim(-xsclip, xsclip)
ax.axhline(y=-ztop, color="gray", linestyle="solid", linewidth=0.5)
ax.axhline(y=-zbot, color="gray", linestyle="solid", linewidth=0.5)
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
ax.set_ylabel("Depth [m]", fontsize=15)
ax.set_title("Shelfbreak jet (1999-2018 XBT average), min valid points = %d"%nobsTi_thresh, fontsize=12)
fig.savefig("CapeHornjet/shelfbreakjet_Tmean.png", bbox_inches="tight", dpi=150)



# Time-averaged ADCP/XBT Ertel PV.
# PV = (f + dudx - dudz*(dTdx/dTdz)) / h
Ts_thresh = 0.15 # dz/dx|T. arctan(dz/dx) = angle of isotherms. dz/dx = 0.15 ~ 8.5 degrees.

dTdz, dTdx = np.gradient(Tsmean)
dz, dx = np.gradient(zT)[:, np.newaxis], np.gradient(xsiT)[np.newaxis, :]*1e3 # [m]
dTdz, dTdx = dTdz/dz, dTdx/dx
# Interpolate temperature gradients to ADCP vertical grid.
Ti = interp_secz(xsiT, zT, Tsmean, z0)
dTdzi = interp_secz(xsiT, zT, dTdz, z0)
dTdxi = interp_secz(xsiT, zT, dTdx, z0)

Ts = dTdxi/dTdzi
Ts[np.abs(Ts)>Ts_thresh] = np.nan
dzu = np.gradient(z0)[:, np.newaxis]
dudz, dudx = np.gradient(usavg)
dudz, dudx = dudz/dzu, dudx/dx
dudx_zavg = np.nanmean(dudx[fzu, :], axis=0)
dudzTs_zavg = np.nanmean(dudz[fzu, :]*Ts[fzu, :], axis=0)

# h = zbot - ztop
T0 = 5.5 # Isotherm that tracks base of pycnocline.
h = get_isoT_depth(z0, Ti, T0)
h = - h

dx = xs[1] - xs[0] # [km]
ya = xs*0
dlat = dx*1/111.12 # km * degree/km
ya[nxiTh:] = latc + dlat*np.arange(0, nxiTh + 1)
ya[:nxiTh] = latc - dlat*np.arange(nxiTh, 0, -1)
fa = fcor(ya).mean()
PV0 = fa/h
PVx = dudx_zavg/h
PVz = - dudzTs_zavg/h
PV = PV0 + PVx + PVz
PVmax = np.nanmax(np.abs(PV[np.abs(xs)<xsclip]))#/2
xsclip2 = 30

fig, ax = plt.subplots()
ax.plot(xs, PV0, "m", linewidth=0.5, label="$f_0/h$")
ax.plot(xs, PVx, "b", linewidth=0.5, label="$u_x/h$")
ax.plot(xs, PVz, "r", linewidth=0.5, label="$-(u_z T_x/T_z)/h$")
ax.plot(xs, PV, "k", linewidth=3, label="Total PV")
ax.set_xlim(-25, 40)
ax.set_ylim(-1.2e-6, 0.3e-6)
ax.axhline(y=0, color="gray", linestyle="dashed")
ax.axvline(x=0, color="gray", linestyle="dashed")
ax.legend(frameon=False, fontsize=16)
ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
ax.set_ylabel("Ertel PV [1/m/s]", fontsize=15)
ax.set_title("Shelfbreak jet (1999-2018 ADCP/XBT average, %d-%d m)"%(ztop, zbot), fontsize=10)
fig.savefig("CapeHornjet/shelfbreakjet_PVmean.png", bbox_inches="tight", dpi=125)


xEDOF = DataArray(np.isfinite(uszavg).sum(axis=0), coords=coordsx, attrs=dict(units="unitless", long_name="Number of effective degrees of freedom based on valid ADCP observations"))
T = DataArray(Tsmean, coords=coordsT, attrs=dict(units="Degrees Celsius", long_name="In situ temperature"))
PV = DataArray(PV, coords=coordsx, attrs=dict(units="1/m/s", long_name="Ertel Potential Vorticity"))
PV0 = DataArray(PV0, coords=coordsx, attrs=dict(units="1/m/s", long_name="Ertel PV thickness term (f/h)"))
PVx = DataArray(PVx, coords=coordsx, attrs=dict(units="1/m/s", long_name="Ertel PV relative vorticity term (v_x/h)"))
PVz = DataArray(PVz, coords=coordsx, attrs=dict(units="1/m/s", long_name="Ertel PV isopycnal slope term [-u_z (T_x/T_z)/h]"))

dvars = dict(us=U, T=T, lon=Lon, lat=Lat, xEDOF=xEDOF, Ldflat=Ldflat, Ldsurf=Ldsurf, PV=PV, PV0=PV0, PVx=PVx, PVz=PVz)
attrs = dict(bounding_isotherm_for_lower_PV_layer="%1.2f degC"%T0)
dsout = Dataset(data_vars=dvars, attrs=attrs).sortby("t")
dsout.to_netcdf("../data/derived/ustream_shelbreakjet_LMG.nc")

plt.show(block=False)
