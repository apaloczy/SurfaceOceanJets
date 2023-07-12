import numpy as np
import matplotlib.pyplot as plt
from xarray import open_dataset

#---
plt.close("all")

suffix = "_fcmaxspd"
head = "../data/derived/"
maxangdiff = 45

dsSAF = open_dataset(head + "ustream_SAFjet_LMG.nc")
dsSAF_spd = open_dataset(head + "ustream_SAFjet_LMG%s.nc"%suffix)
dsPF = open_dataset(head + "ustream_PFjet_LMG.nc")
dsPF_spd = open_dataset(head + "ustream_PFjet_LMG%s.nc"%suffix)
dsSACCF = open_dataset(head + "ustream_SACCFjet_LMG.nc")
dsSACCF_spd = open_dataset(head + "ustream_SACCFjet_LMG%s.nc"%suffix)

usSAF = dsSAF.where(np.abs(dsSAF["angdiff"])<maxangdiff)["us"]
usSAF_spd = dsSAF_spd.where(np.abs(dsSAF_spd["angdiff"])<maxangdiff)["us"]
usPF = dsPF.where(np.abs(dsPF["angdiff"])<maxangdiff)["us"]
usPF_spd = dsPF_spd.where(np.abs(dsPF_spd["angdiff"])<maxangdiff)["us"]
usSACCF = dsSACCF.where(np.abs(dsSACCF["angdiff"])<maxangdiff)["us"]
usSACCF_spd = dsSACCF_spd.where(np.abs(dsSACCF_spd["angdiff"])<maxangdiff)["us"]

usSAF_SE = usSAF.std("t")/np.sqrt(np.isfinite(usSAF.values).sum(axis=0))
usSAF_spd_SE = usSAF_spd.std("t")/np.sqrt(np.isfinite(usSAF_spd.values).sum(axis=0))
usPF_SE = usPF.std("t")/np.sqrt(np.isfinite(usPF.values).sum(axis=0))
usPF_spd_SE = usPF_spd.std("t")/np.sqrt(np.isfinite(usPF_spd.values).sum(axis=0))
usSACCF_SE = usSACCF.std("t")/np.sqrt(np.isfinite(usSACCF.values).sum(axis=0))
usSACCF_spd_SE = usSACCF_spd.std("t")/np.sqrt(np.isfinite(usSACCF_spd.values).sum(axis=0))

usSAF = usSAF.mean("t")
usSAF_spd = usSAF_spd.mean("t")
usPF = usPF.mean("t")
usPF_spd = usPF_spd.mean("t")
usSACCF = usSACCF.mean("t")
usSACCF_spd = usSACCF_spd.mean("t")

usSAF_CL95l, usSAF_CL95u = usSAF - 2*usSAF_SE, usSAF + 2*usSAF_SE
usSAF_spd_CL95l, usSAF_spd_CL95u = usSAF_spd - 2*usSAF_spd_SE, usSAF_spd + 2*usSAF_spd_SE
usPF_CL95l, usPF_CL95u = usPF - 2*usPF_SE, usPF + 2*usPF_SE
usPF_spd_CL95l, usPF_spd_CL95u = usPF_spd - 2*usPF_spd_SE, usPF_spd + 2*usPF_spd_SE
usSACCF_CL95l, usSACCF_CL95u = usSACCF - 2*usSACCF_SE, usSACCF + 2*usSACCF_SE
usSACCF_spd_CL95l, usSACCF_spd_CL95u = usSACCF_spd - 2*usSACCF_spd_SE, usSACCF_spd + 2*usSACCF_spd_SE

xs = dsSAF["x"].values
xmin, xmax = xs[0], xs[-1]
ymin, ymax = -0.1, 0.8

fig, ax = plt.subplots(figsize=(10, 8))

ax.fill_between(xs, usSAF_CL95l, usSAF_CL95u, color="r", alpha=0.1)
ax.fill_between(xs, usSAF_spd_CL95l, usSAF_spd_CL95u, color="r", alpha=0.1)
ax.fill_between(xs, usPF_CL95l, usPF_CL95u, color="y", alpha=0.1)
ax.fill_between(xs, usPF_spd_CL95l, usPF_spd_CL95u, color="y", alpha=0.1)
ax.fill_between(xs, usSACCF_CL95l, usSACCF_CL95u, color="b", alpha=0.1)
ax.fill_between(xs, usSACCF_spd_CL95l, usSACCF_spd_CL95u, color="b", alpha=0.1)

usSAF_spd.plot(c="r", ls="dashed", label="SAF, max($\sqrt{u^2 + v^2}$)")
usSAF.plot(c="r", ls="solid", label="SAF, max($u$)")
usPF_spd.plot(c="y", ls="dashed", label="PF, max($\sqrt{u^2 + v^2}$)")
usPF.plot(c="y", ls="solid", label="PF, max($u$)")
usSACCF_spd.plot(c="b", ls="dashed", label="SACCF, max($\sqrt{u^2 + v^2}$)")
usSACCF.plot(c="b", ls="solid", label="SACCF, max($u$)")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axhline(color="gray")
ax.axvline(color="gray")
ax.set_xlabel("Cross-stream distance, $x$ [km]", fontsize=15, fontweight="black")
ax.set_ylabel("Downstream velocity, $v$ [m/s]", fontsize=15, fontweight="black")
ax.legend(ncol=1)
fig.savefig("ACCjets/cmp_ACCjets_maxu_maxspd.png", bbox_inches="tight")

plt.show(block=False)
