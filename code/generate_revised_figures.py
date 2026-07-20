#!/usr/bin/env python3
"""Generate revised/new figures and compute REAL derived statistics.
No data are fabricated: every numeric value is taken from the manuscript /
notebook execution outputs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = str(ROOT / "figures")
OUT = str(ROOT / "results")
Path(FIG).mkdir(parents=True, exist_ok=True)
Path(OUT).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.dpi": 300})

# 1. REAL 95% CIs from five per-fold CV values -------------------------
folds = {"Accuracy":[0.6876,0.7271,0.6352,0.7322,0.7309],
         "Macro-F1":[0.5221,0.5155,0.4316,0.5642,0.5694],
         "QWK":[0.7359,0.8294,0.7619,0.8223,0.8172],
         "AUC":[0.8775,0.8815,0.8654,0.8999,0.9066],
         "ECE":[0.2424,0.1820,0.1190,0.1718,0.1738]}
n=5; tcrit=stats.t.ppf(0.975,df=n-1)
print("=== 95%% CI from 5-fold CV (t, df=4, tcrit=%.4f) ==="%tcrit)
with open(OUT+"/ci_stats.txt","w") as f:
    for k,v in folds.items():
        v=np.array(v); m,sd=v.mean(),v.std(ddof=1); hw=tcrit*sd/np.sqrt(n)
        line="%-9s: %.4f +/- %.4f (SD) | 95%% CI [%.4f, %.4f] (+/-%.4f)"%(k,m,sd,m-hw,m+hw,hw)
        print(line); f.write(line+"\n")

def box(ax,x,y,w,h,text,fc,ec="#222",fs=9,tc="#111",lw=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=tc,zorder=3)
def arrow(ax,x1,y1,x2,y2,color="#444",lw=1.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,color=color,lw=lw,zorder=1))

# 2. REDRAWN Fig 1 architecture ---------------------------------------
fig,ax=plt.subplots(figsize=(13,6.2)); ax.set_xlim(0,13); ax.set_ylim(-0.3,6.2); ax.axis("off")
C_IN="#E3F2FD";C_BB="#BBDEFB";C_ATT="#C8E6C9";C_FUSE="#FFE0B2";C_SSM="#D1C4E9";C_CLS="#FFCDD2";C_REL="#F8BBD0";C_LOSS="#ECEFF1"
box(ax,0.1,2.6,1.4,1.0,"Input\n224x224x3\nGaussian-filt.\n+ CLAHE",C_IN,fs=7.5); arrow(ax,1.5,3.1,2.0,3.1)
box(ax,2.0,1.5,1.6,3.2,"EfficientNet-B0\n(Noisy-Student)\nfully fine-tuned",C_BB,fs=8)
fm=[("F3 28x28x40",4.0,3.75),("F4 14x14x112",4.0,2.55),("F5 7x7x320",4.0,1.35)]
for t,x,y in fm:
    box(ax,x,y,1.4,0.7,t,"#E1F5FE",fs=7.3); arrow(ax,3.6,3.1,x-0.02,y+0.35)
    box(ax,5.55,y,1.1,0.7,"CBAM",C_ATT,fs=8); arrow(ax,x+1.4,y+0.35,5.55,y+0.35)
    box(ax,6.75,y,1.2,0.7,"1x1->128\n+upsample",C_ATT,fs=6.6); arrow(ax,6.65,y+0.35,6.75,y+0.35)
    arrow(ax,7.95,y+0.35,8.35,2.9)
box(ax,8.35,2.2,1.4,1.4,"Concat +\n3x3 Fusion\nBN+GELU\n256 ch",C_FUSE,fs=7.2); arrow(ax,9.75,2.9,10.15,2.9)
box(ax,10.15,2.35,1.3,1.05,"Tokenize\n784 tok (256d)",C_SSM,fs=7.0); arrow(ax,10.8,2.35,10.8,1.95)
box(ax,10.15,1.15,1.3,0.75,"2x Gated\nSSM blocks",C_SSM,fs=7.2)
ax.text(10.8,0.72,"DWConv1D(k=5)+sigmoid gate+residual",ha="center",fontsize=5.9,color="#555")
arrow(ax,11.45,1.5,11.9,1.5); arrow(ax,11.7,2.9,11.9,2.9)
box(ax,11.9,1.15,1.0,2.3,"Mean-\npool\nz-bar\n(256)",C_SSM,fs=7.2)
arrow(ax,12.4,3.45,12.4,3.9); box(ax,11.55,3.9,1.4,0.8,"Cls head\n-> 5 grades",C_CLS,fs=7.3)
arrow(ax,12.4,1.15,12.4,0.72); box(ax,11.55,-0.05,1.4,0.75,"Reliability\nhead c in[0,1]",C_REL,fs=7.0)
box(ax,2.0,5.15,9.4,0.8,r"Objective  $\mathcal{L}=\mathcal{L}_{focal}(\gamma{=}2)+\lambda\mathcal{L}_{cal}(\lambda{=}0.1)+\beta\mathcal{L}_{rel}(\beta{=}0.5,\ \mathrm{self\text{-}reflective\ BCE})$",C_LOSS,fs=10)
ax.text(6.7,5.99,"ABSS-Net architecture",ha="center",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig(FIG+"/fig1_architecture.png",bbox_inches="tight",dpi=300,facecolor="white")
plt.savefig(FIG+"/fig1_architecture.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("fig1_architecture redrawn")

# 3. SIMPLIFIED Fig 2 --------------------------------------------------
classes=["No_DR","Mild","Moderate","Severe","Prolif."]; counts=[1805,370,999,193,295]
colors=["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0"]; total=sum(counts)
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={"width_ratios":[1.5,1]})
bars=a1.bar(classes,counts,color=colors,edgecolor="#333",linewidth=0.7)
for b,c in zip(bars,counts):
    a1.text(b.get_x()+b.get_width()/2,c+20,"%d\n(%.1f%%)"%(c,100*c/total),ha="center",va="bottom",fontsize=8.3)
a1.set_ylabel("Number of images"); a1.set_title("(a) Class frequency",fontweight="bold")
a1.set_ylim(0,2050); a1.tick_params(axis="x",rotation=15)
for s in ["top","right"]: a1.spines[s].set_visible(False)
imb=max(counts)/min(counts)
a2.pie(counts,colors=colors,autopct=lambda p:"%.0f%%"%p,startangle=90,
       wedgeprops=dict(edgecolor="white",linewidth=1),textprops={"fontsize":8})
a2.set_title("(b) Proportion (imbalance %.1f:1)"%imb,fontweight="bold")
a2.legend(classes,loc="center left",bbox_to_anchor=(0.98,0.5),fontsize=7.5,frameon=False)
plt.tight_layout(); plt.savefig(FIG+"/fig2_class_distribution.png",bbox_inches="tight",dpi=300,facecolor="white"); plt.close(fig)
print("fig2_class_distribution simplified")

# 4. NEW reliability c-curve (R2.3) -----------------------------------
cmid=np.array([0.1,0.3,0.5,0.7,0.9]); acc=np.array([0.5000,0.4591,0.4444,0.4899,0.9559]); ns=np.array([6,562,630,694,1770])
fig,ax=plt.subplots(figsize=(6.6,5))
ax.plot([0,1],[0,1],"--",color="#9E9E9E",lw=1.2,label="Perfect calibration (acc = c)")
ax.plot(cmid,acc,"-",color="#7C4DFF",lw=2,zorder=2)
ax.scatter(cmid,acc,s=40+260*ns/ns.max(),c="#7C4DFF",edgecolor="#311B92",zorder=3,alpha=0.9)
for x,y,nn in zip(cmid,acc,ns):
    ax.annotate("n=%d\nacc=%.3f"%(nn,y),(x,y),textcoords="offset points",xytext=(8,-16 if y>0.6 else 8),fontsize=7.3)
ax.axvline(0.76,color="#F44336",ls=":",lw=1.3)
ax.text(0.775,0.03,r"threshold $c^\ast=0.76$",color="#F44336",fontsize=8,rotation=90,va="bottom")
ax.set_xlabel("Reliability score  c  (bin midpoint)"); ax.set_ylabel("Empirical accuracy of retained samples")
ax.set_title("Reliability-head calibration curve (pooled OOF)\nmarker area proportional to bin count",fontweight="bold",fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3); ax.legend(loc="upper left",fontsize=8)
plt.tight_layout(); plt.savefig(FIG+"/fig19_reliability_c_curve.png",bbox_inches="tight",dpi=300,facecolor="white"); plt.close(fig)
print("fig19_reliability_c_curve created (NEW)")
print("ALL FIGURES DONE")
