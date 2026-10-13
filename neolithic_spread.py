"""
Real case study: the spread of farming into Europe (Neolithic transition)
compared with the Fisher-KPP front speed.

Data: Table S1 of
    Pinhasi R, Fort J, Ammerman AJ (2005). Tracing the origin and spread of
    agriculture in Europe. PLoS Biology 3(12): e410. doi:10.1371/journal.pbio.0030410
(open access, Creative Commons Attribution licence), saved as
data/pinhasi2005_neolithic_sites.csv: 765 early Neolithic sites with their
earliest radiocarbon date and the distance to the centre Abu Madi (Sinai).

1. Observed speed: as in the paper, regress distance on date and date on
   distance for the 735 sites of Europe, Anatolia and the Near East
   (the 30 Arabian sites were only a side test in the paper).
2. Predicted speed, with the parameters of the paper:
       a = 0.029 .. 0.035 /year      initial growth rate
       m = 900 .. 2200 km^2          mobility (mean squared distance per generation)
       T = 29 .. 35 years            generation time
       D = m / (4 T)                 diffusion coefficient
   Fisher-KPP:            v = 2 sqrt(a D)
   time-delayed Fisher:   v = 2 sqrt(a D) / (1 + a T / 2)    (Fort & Mendez 1999)

Run:  python neolithic_spread.py  -> FigN1_neolithic_data.png, FigN2_neolithic_speed.png
"""

import csv
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "data/pinhasi2005_neolithic_sites.csv"
ABU_MADI = (28.55, 33.97)                     # (lat, lon) of the centre

# parameter ranges from Pinhasi et al. (2005)
A_RANGE = (0.029, 0.035)                      # 1/year
M_RANGE = (900.0, 2200.0)                     # km^2 per generation
T_RANGE = (29.0, 35.0)                        # years
OBSERVED_PAPER = (0.6, 1.3)                   # km/year, all centres and methods (95%)

# ============================================================
# 1. DATA AND OBSERVED SPEED
# ============================================================

with open(DATA_FILE, encoding="utf-8") as f:
    rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
sites = [r for r in rows if r["region"] != "Arabia" and r["uncal_bp"]]

lat = np.array([float(r["lat"]) for r in sites])
lon = np.array([float(r["lon"]) for r in sites])
date = np.array([float(r["uncal_bp"]) for r in sites])          # years before present
dist = np.array([float(r["dist_gc_abu_madi_km"]) for r in sites])

R = abs(np.corrcoef(date, dist)[0, 1])
slope_d_t, icpt_d_t = np.polyfit(date, dist, 1)     # distance = slope * date + icpt
slope_t_d, icpt_t_d = np.polyfit(dist, date, 1)     # date = slope * distance + icpt
v_low = -slope_d_t                                  # km per year
v_high = -1.0 / slope_t_d

print(f"sites: {len(sites)},  dates {date.max():.0f} .. {date.min():.0f} BP (uncalibrated)")
print(f"correlation R = {R:.3f}")
print(f"observed speed: distance-vs-date {v_low:.2f} km/yr, date-vs-distance {v_high:.2f} km/yr")

# ============================================================
# 2. PREDICTED SPEED
# ============================================================


def fisher(a, m, T):
    return 2 * math.sqrt(a * m / (4 * T))


def delayed(a, m, T):
    return fisher(a, m, T) / (1 + a * T / 2)


corners = list(itertools.product(A_RANGE, M_RANGE, T_RANGE))
v_fisher = (min(fisher(*c) for c in corners), max(fisher(*c) for c in corners))
v_delay = (min(delayed(*c) for c in corners), max(delayed(*c) for c in corners))
D_range = (M_RANGE[0] / (4 * T_RANGE[1]), M_RANGE[1] / (4 * T_RANGE[0]))
print(f"D = m/(4T) = {D_range[0]:.1f} .. {D_range[1]:.1f} km^2/yr")
print(f"Fisher-KPP:          v = {v_fisher[0]:.2f} .. {v_fisher[1]:.2f} km/yr")
print(f"time-delayed Fisher: v = {v_delay[0]:.2f} .. {v_delay[1]:.2f} km/yr")

# ============================================================
# 3. FIGURES (sized for a full slide width)
# ============================================================

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888",
                     "axes.labelcolor": "#222222", "xtick.color": "#555555",
                     "ytick.color": "#555555"})
RED, BLUE, GREY = "#A02020", "#2B6CB0", "#555555"

# ---------- FigN1: the data ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.15, 1]})

order = np.argsort(-date)                            # draw the oldest first
sc = ax1.scatter(lon[order], lat[order], c=date[order], cmap="YlOrBr", s=9,
                 edgecolors="#555555", linewidths=0.2)
ax1.plot(ABU_MADI[1], ABU_MADI[0], marker="*", ms=12, color="white", mec="black", mew=0.9)
ax1.annotate("Abu Madi", (ABU_MADI[1], ABU_MADI[0]), textcoords="offset points",
             xytext=(8, -1), ha="left", va="center", fontsize=8, color="#222222")
ax1.set_aspect(1 / math.cos(math.radians(42)))
ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")
ax1.set_title(f"{len(sites)} early Neolithic sites", fontsize=11)
fig.colorbar(sc, ax=ax1, shrink=0.85).set_label("earliest date (years BP)")

ax2.scatter(dist, date, s=7, color="#bbbbbb", edgecolors="#777777", linewidths=0.2)
d_line = np.linspace(0, dist.max(), 2)
ax2.plot(d_line, (d_line - icpt_d_t) / slope_d_t, color=RED, linewidth=2,
         label=f"fit d(t): {v_low:.2f} km/yr")
ax2.plot(d_line, slope_t_d * d_line + icpt_t_d, color=BLUE, linewidth=2,
         label=f"fit t(d): {v_high:.2f} km/yr")
ax2.invert_yaxis()                                   # older at the bottom
ax2.set_xlabel("Distance from Abu Madi (km)")
ax2.set_ylabel("Date (years BP)")
ax2.set_title(f"Date vs distance  (R = {R:.3f})", fontsize=11)
ax2.legend(frameon=False, fontsize=8, loc="upper left")
ax2.grid(color="#e6e6e6", linewidth=0.8)
ax2.spines[["top", "right"]].set_visible(False)

fig.savefig("FigN1_neolithic_data.png", dpi=250)
plt.close(fig)
print("saved FigN1_neolithic_data.png")

# ---------- FigN2: observed vs predicted speed ----------
fig, ax = plt.subplots(figsize=(7.2, 2.3), constrained_layout=True)

ax.axvspan(*OBSERVED_PAPER, color="#eeeeee", zorder=0)
bars = [
    ("Observed (paper, 95%)", OBSERVED_PAPER, GREY),
    ("Observed (our refit)", (v_low, v_high), GREY),
    ("Fisher–KPP  $2\\sqrt{aD}$", v_fisher, RED),
    ("Time-delayed Fisher", v_delay, BLUE),
]
for k, (label, (lo, hi), color) in enumerate(bars):
    y = len(bars) - 1 - k
    ax.plot([lo, hi], [y, y], color=color, linewidth=7, solid_capstyle="butt")
    ax.text(hi + 0.03, y, f"{lo:.2f} – {hi:.2f}", va="center", fontsize=9, color="#222222")
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in reversed(bars)])
ax.set_xlim(0.4, 2.0)
ax.set_ylim(-0.6, len(bars) - 0.4)
ax.set_xlabel("Speed of the farming front (km/year)")
ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.savefig("FigN2_neolithic_speed.png", dpi=250)
plt.close(fig)
print("saved FigN2_neolithic_speed.png")
