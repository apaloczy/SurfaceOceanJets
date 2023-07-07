# Convert all the jet profiles to the same format for Figures 2 and 3.
import numpy as np
import matplotlib.pyplot as plt
from xarray import open_dataset
from scipy.optimize import curve_fit
from glob import glob


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


def near(x, x0):
    return np.nanargmin(np.abs(x - x0))


def nearfirst_decreasing(u, u0):
    """
    Returns index of first element whose value is below `u0`
    for a decreasing array `u`.
    """
    return u[np.where(u - u0 <= 0)[0][0]]

def getLjfs(xsaux, usaux, usauxl, usauxr, uscapL):
    try:
        usfrL = nearfirst_decreasing(usauxr, uscapL)
        Ljrf = xsaux[usaux==usfrL][0]
    except IndexError:
        Ljrf = np.nan

    try:
        usflL = nearfirst_decreasing(np.flipud(usauxl), uscapL)
        Ljlf = xsaux[usaux==usflL][0]
    except IndexError:
        Ljlf = np.nan

    return Ljlf, Ljrf


def fitfunc(x, a, b): # with the constraint that the peak velocity matches that in the observed profile.
    return a*x + b


def exfit_Lj_singleexp(x, u, xlrtrim, umin=0.05, uctol=0.001, CAP_CENTER=True, CORE_CONSTRAINT=True):
    xtriml, xtrimr = xlrtrim
    ftl, ftr = near(x, xtriml), near(x, xtrimr)
    if ftr < (x.size - 1):
        ftr += 1
    trim = slice(ftl, ftr + 1)
    x0 = x.copy()
    x = x[trim]
    u = u[trim]

    nxh = np.where(x==0)[0][0] # Get left and right sides of the jet.
    if CAP_CENTER:
        nxhl, nxhr = nxh, nxh + 1 # Skip u(x=0).
    else:
        nxhl, nxhr = nxh + 1, nxh # Include u(x=0).

    fl, fr = slice(0, nxhl), slice(nxhr, None) # Keep origin on both sides.
    xl, xr = x[fl], x[fr]
    ul, ur = u[fl], np.flipud(u[fr])
    xl = xl - xl[0]
    xr = np.flipud(xr[-1] - xr)

    # Add offsets to do the fits.
    uoffl = -np.nanmin(ul) + umin
    uoffr = -np.nanmin(ur) + umin

    ul = ul + uoffl
    ur = ur + uoffr

    # Perform a linear fit in log space on each side.
    fgl = np.isfinite(ul)
    fgr = np.isfinite(ur)
    if CORE_CONSTRAINT: # Least-squares fit a line to the profile in log space, with a constraint on the bounds of the core velocity of the resulting exponential.
        ucore = np.nanmax(u)
        ucorel = ucore + uoffl
        ucorer = ucore + uoffr

        xll, xrr = xl[fgl], xr[fgr]

        # Shift origin to final point. Now the intercept is the core velocity.
        xll = xll - xll[-1]
        xrr = xrr - xrr[-1]
        ull, urr = np.log(ul[fgl]), np.log(ur[fgr])
        abguessl = ((ull[-1] - ull[0])/(xll[-1] - xll[0]), np.log(ucorel))
        abguessr = ((urr[-1] - urr[0])/(xrr[-1] - xrr[0]), np.log(ucorer))
        bndsl = ((0, np.log(ucorel - uctol)), (np.inf, np.log(ucorel + uctol)))
        bndsr = ((0, np.log(ucorer - uctol)), (np.inf, np.log(ucorer + uctol)))

        maxfev = 10000
        (al, bl), _ = curve_fit(fitfunc, xll, ull, p0=abguessl, bounds=bndsl, maxfev=maxfev)
        (ar, br), _ = curve_fit(fitfunc, xrr, urr, p0=abguessr, bounds=bndsr, maxfev=maxfev)

        # y arrays ready to plot over the full range.
        x0l = x0
        x0r = - np.flipud(x0)
        ypl = np.exp(al*x0l + bl) - uoffl
        ypr = np.flipud(np.exp(ar*x0r + br)) - uoffr
    else: # Least-squares fit a line to the profile in log space, no constraints.
        al, bl = np.polyfit(xl[fgl], np.log(ul[fgl]), 1)
        ar, br = np.polyfit(xr[fgr], np.log(ur[fgr]), 1)

        # x and y arrays ready to plot over the full range.
        x0l = x0 - x0[trim.start]
        x0r = - np.flipud(x0 - x0[trim.stop])
        ypl = np.exp(al*x0l + bl) - uoffl
        ypr = np.flipud(np.exp(ar*x0r + br)) - uoffr

    return 1/al, 1/ar, ypl, ypr


#---
plt.close("all")

CORE_CONSTRAINT = True
CAP_CENTER = False
out_prefix = "../data/derived/"

umaxfracLs = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
umaxfrac = 0.10 # Fraction of the core velocity to cap the fit at either side of the jet.

AVGLR_ufrac = True
umaxfrac_Ldlr = 0.4
nLdcap_avg = 1
ACCjets_angdiff_thresh = 45 # Maximum instantaneous stream rotation angle to include in time-average for the ADCP data [Degrees]

names = ["GulfStream33N", "GulfStream34N", "GulfStream36N", "GulfStream37N", "GulfStream38N", "AgulhasCurrent", "EAC29S", "BrazilCurrent29S", "KuroshioCurrent25N", "KuroshioCurrent28p5N", "ustream_SAFjet_LMG.nc", "ustream_PFjet_LMG.nc", "ustream_SACCFjet_LMG.nc", "ustream_shelbreakjet_LMG.nc"]

# Add lower-latitude jets from synoptic ADCP sections.
names_synop = glob("%s*_synop.nc"%out_prefix)
names_synop = [n.removeprefix(out_prefix) for n in names_synop]
names.extend(names_synop)

if CAP_CENTER:
    cap_center = "yescappedcore"
else:
    cap_center = "nocappedcore"

if CORE_CONSTRAINT:
    core_constraint = "boundedcore"
else:
    core_constraint = "unboundedcore"

ACCnames = ["SAFjet", "PFjet", "SACCFjet"]
for name in names:
    print(name, "================================")
    if "synop" in name:
        SYNOP = True
    else:
        SYNOP = False

    if "GulfStream" in name or name in ["AgulhasCurrent"]:
        nameh = name
    elif SYNOP:
        f = name
    elif "shelbreakjet_LMG" in name:
        f = name
        name = "LMGshfbrk"
    elif "SAFjet" in name:
        f = name
        name = "SAFjet"
    elif "PFjet" in name:
        f = name
        name = "PFjet"
    elif "SACCFjet" in name:
        f = name
        name = "SACCFjet"
    elif name=="EAC29S":
        nameh = "EastAustralianCurrent29S"
    else:
        nameh = name

    if "LMGshfbrk" not in name and name not in ACCnames and not SYNOP:
        f = "%s/ug_stream_TPJAOS-%s.nc"%(nameh, name)
    else:
        f = out_prefix + f

    ds = open_dataset(f)
    if SYNOP:
        xs = ds["x"].values
        us = ds["u"].values
    elif name in ACCnames or "LMGshfbrk" in name:
        xs = ds["x"].values
        if "LMGshfbrk" in name:
            fangg = np.abs(ds["angcr"])<=ACCjets_angdiff_thresh
        else:
            fangg = np.abs(ds["angdiff"])<=ACCjets_angdiff_thresh
        us = ds["us"].where(fangg)
        pg = 100*np.sum(fangg.values)/fangg.size
        print("Kept %d%% of values for %s"%(pg, name))
        xEDOF = np.isfinite(us).sum(dim="t")
        SE = us.std(dim="t")/np.sqrt(xEDOF)
        us = us.mean(dim="t").values
        us[np.abs(xs)>50] = np.nan
    else:
        xs = ds["x"].values
        xEDOF = ds["xEDOF"].values
        SE = ds["us"].std("t").values/np.sqrt(xEDOF)
        us = ds["us"].mean("t").values

    if name in ACCnames or name=="LMGshfbrk":
        Ld_surf, Ld_flat = ds["Ldsurf"].mean("t").values, ds["Ldflat"].mean("t").values
    else:
        Ld_surf, Ld_flat = ds["Ldsurf"].values, ds["Ldflat"].values

    if name in ACCnames or name=="LMGshfbrk" or SYNOP:
        IS_ALTIMETRY = False
    else:
        IS_ALTIMETRY = True

    if name=="EAC29S":
        dCL95thresh = 0.3
    elif name=="KuroshioCurrent25N":
        dCL95thresh = 0.2
    else:
        dCL95thresh = 0.15

    if not SYNOP:
        CL95l = us - 2*SE
        CL95u = us + 2*SE
        dCL95 = CL95u - CL95l
        us[dCL95>dCL95thresh] = np.nan

    fcavg = np.nanargmax(us)
    nxh = np.where(xs==0)[0][0]
    if fcavg!=nxh:
        nshft = nxh - fcavg
        us = rolltrim(us, nshft)
        Ld_surf = rolltrim(Ld_surf, nshft)
        Ld_flat = rolltrim(Ld_flat, nshft)
        if not SYNOP and name not in ACCnames:
            CL95l = rolltrim(CL95l, nshft)
            CL95u = rolltrim(CL95u, nshft)
            dCL95 = rolltrim(dCL95, nshft)
        print("")
        print("Shifted averaged ug profile by %d point(s)."%np.abs(nshft))
        print("")

    if SYNOP:
        name = name.split("/")[-1].split("_synop")[0]

    usmax = np.nanmax(us)
    uscap = usmax*umaxfrac
    usauxl, usauxr = us.copy(), us.copy()
    usauxl[nxh:] = np.nan
    usauxr[:nxh] = np.nan

    if ~np.any((usauxl - uscap)<=0):
        usfl = np.nanmin(usauxl)
    else:
        usfl = nearfirst_decreasing(np.flipud(usauxl), uscap)

    if ~np.any((usauxr - uscap)<=0):
        usfr = np.nanmin(usauxr)
    else:
        usfr = nearfirst_decreasing(usauxr, uscap)

    xltrim, xrtrim = xs[us==usfl][0], xs[us==usfr][0]
    if name=="LMGshfbrk":
        xrtrim = 25 # Approximately 25% umax on the right side.

    xlrtrim = (xltrim, xrtrim)

    # Also find the position of a fractional drop-off in velocity to compare with Ld.
    # Make an interpolated jet profile just to get better reolution on the cutoff points.
    xsmin = -np.maximum(-xs[0], xs[-1])
    xsaux = np.linspace(xsmin, 0, num=1000)
    xsaux = np.hstack((xsaux, np.flipud(-xsaux)[1:]))
    usaux = np.interp(xsaux, xs, us, left=np.nan, right=np.nan)

    usauxl, usauxr = usaux.copy(), usaux.copy()
    nxhaux = np.where(xsaux==0)[0][0]
    usauxl[nxhaux:] = np.nan
    usauxr[:nxhaux] = np.nan

    print("")
    Ljlfs, Ljrfs = dict(), dict()
    for umaxfracL in umaxfracLs:
        uscapL = usmax*umaxfracL
        Ljlfi, Ljrfi = getLjfs(xsaux, usaux, usauxl, usauxr, uscapL)
        Ljlfs.update({umaxfracL:Ljlfi})
        Ljrfs.update({umaxfracL:Ljrfi})
        print("%d%% umax left/right: %1.1f"%(umaxfracL*100, Ljlfi), "/", "%1.1f km"%Ljrfi)
    print("")

    Ljl, Ljr, ypl, ypr = exfit_Lj_singleexp(xs, us, xlrtrim, umin=0.05, uctol=0.001, CAP_CENTER=CAP_CENTER, CORE_CONSTRAINT=CORE_CONSTRAINT)

    xlu, xru = Ljlfs[umaxfrac_Ldlr], Ljrfs[umaxfrac_Ldlr]
    if AVGLR_ufrac:
        Ll, Lr = xsaux[near(xsaux, xlu)], xs[near(xs, xru)]
        fl, fr = np.logical_and(xs>Ll, xs<0), np.logical_and(xs>0, xs<Lr)
        Ldl_surf, Ldr_surf = np.nanmean(Ld_surf[fl]), np.nanmean(Ld_surf[fr])
        Ldl_flat, Ldr_flat = np.nanmean(Ld_flat[fl]), np.nanmean(Ld_flat[fr])
    else:
        fl, fr = near(xsaux, xlu), near(xsaux, xru)
        Ldl_surf, Ldr_surf = Ld_surf[fl], Ld_surf[fr]
        Ldl_flat, Ldr_flat = Ld_flat[fl], Ld_flat[fr]

    if not SYNOP:
        eke = ((ds["us"] - ds["us"].mean("t"))**2).mean("t")/2
        eke[dCL95>dCL95thresh] = np.nan
        fcapeke = np.logical_or(xs<-Ldl_surf*nLdcap_avg, xs>Ldr_surf*nLdcap_avg)
        eke[fcapeke] = np.nan
        ekeavg = np.nanmean(eke)

    cl, cr = "r", "b"
    cl2, cr2 = "r--", "b--"

    if name in ["LMGshfbrk"]:
        xsclip = 30
    else:
        xsclip = 140

    # Save Lj, Ld values for scatterplot.
    if ds["lat"].ndim==1:
        lonn, latt = ds["lon"], ds["lat"]
    else:
        print("*************** ", name, " averaging lon, lat in time")
        lonn, latt = ds["lon"].mean("t"), ds["lat"].mean("t")

    fm = np.where(xs==0)[0][0]
    lon0, lat0 = lonn.values[fm], latt.values[fm]

    npzout = dict(x=xs, us=us, lon=lonn, lat=latt, lon0=lon0, lat0=lat0, CL95l=CL95l, CL95u=CL95u, ypl=ypl, ypr=ypr, Ljl=Ljl, Ljr=Ljr, Ljlufrac=Ljlfs, Ljrufrac=Ljrfs, xlt=xltrim, xrt=xrtrim, Ldlsurf=Ldl_surf, Ldrsurf=Ldr_surf, Ldlflat=Ldl_flat, Ldrflat=Ldr_flat)
    if IS_ALTIMETRY:
        vADT = ds["vADTs"].mean("t").values
        npzout.update(dict(vADT=vADT))

    if not SYNOP:
        npzout.update(dict(ekeavg=ekeavg))

    np.savez(out_prefix + name + ".npz", **npzout)

    dyl, dyt = 0.1, 0.18
    fig, ax = plt.subplots()
    ax.axhline(y=0, color="gray", linestyle="dashed")
    ax.axvline(x=0, color="gray", linestyle="dashed")
    if not SYNOP:
        ax.fill_between(xs, CL95l, CL95u, color="k", alpha=0.2)
    ax.plot(xs, us, "k")
    ax.set_xlim(-xsclip, xsclip)
    ax.set_ylim(np.nanmin(us) - dyl, np.nanmax(us) + dyl)
    ax.text(0.2, 0.75+dyt, "$L_{dl}$ = %.1f, %.1f km"%(Ldl_surf, Ldl_flat), fontsize=12, transform=ax.transAxes, ha="center", color="r")
    ax.text(0.8, 0.75+dyt, "$L_{dr}$ = %.1f, %.1f km"%(Ldr_surf, Ldr_flat), fontsize=12, transform=ax.transAxes, ha="center", color="b")
    ax.axvline(xlrtrim[0], color=cl, alpha=0.2)
    ax.axvline(xlrtrim[1], color=cr, alpha=0.2)
    ax.plot(xs, ypl, cl)
    ax.plot(xs, ypr, cr)
    for umaxfracL in umaxfracLs:
        Ljlfi, Ljrfi = Ljlfs[umaxfracL], Ljrfs[umaxfracL]
        if np.isfinite(Ljlfi):
            ax.plot(Ljlfi, usaux[xsaux==Ljlfi], marker="o", mfc="b", mec="b")
        if np.isfinite(Ljrfi):
            ax.plot(Ljrfi, usaux[xsaux==Ljrfi], marker="o", mfc="b", mec="b")

    ax.text(0.2, 0.68+dyt, "$e_l$ = %.1f km"%Ljl, fontsize=12, transform=ax.transAxes, ha="center", color="r")
    ax.text(0.8, 0.68+dyt, "$e_r$ = %.1f km"%Ljr, fontsize=12, transform=ax.transAxes, ha="center", color="b")
    ax.text(0.2, 0.61+dyt, "$e_l/L_{dl}$ = %.1f, %.1f"%(Ljl/Ldl_surf, Ljl/Ldl_flat), fontsize=12, transform=ax.transAxes, ha="center", color="r")
    ax.text(0.8, 0.61+dyt, "$e_l/L_{dr}$ = %.1f, %.1f"%(Ljr/Ldr_surf, Ljr/Ldr_flat), fontsize=12, transform=ax.transAxes, ha="center", color="b")
    ax.set_xlabel("Cross-stream distance [km]", fontsize=15)
    ax.set_ylabel("Downstream velocity $v$ [m/s]", fontsize=15)
    ax.set_title("%s (along-track altimetry), cap at $\pm$%1.2f umax"%(name, umaxfrac), fontsize=10)
    umaxfracstr = str(umaxfrac).replace(".", "p") + "umax"
    figname = "expfit/jet_umean_%s_%s_%s_%s.png"%(name, umaxfracstr, cap_center, core_constraint)
    fig.savefig(figname, bbox_inches="tight", dpi=125)
