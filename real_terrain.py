"""
Real terrain of a region from a digital elevation model (DEM), with rivers.

  1. download Copernicus DEM GLO-30 tiles (about 30 m resolution, free, no login)
  2. download river centre lines from OpenStreetMap (Overpass API)
  3. crop the region and convert longitude / latitude to kilometres
  4. topographic map: elevation tints + hillshade + contour lines + rivers
  5. 2D vertical cross-section (terrain profile) along a line A-B,
     with the points where A-B crosses a river

Regions are defined in REGIONS below; add your own the same way
(any region between 60S and 60N works).

Data: Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and
Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
Rivers: (c) OpenStreetMap contributors, ODbL.

Run:  python real_terrain.py            -> default region (nhoque)
      python real_terrain.py bavi       -> another region
      -> FigR1_topo_map_<region>.png, FigR2_profile_<region>.png, data/dem_<region>.npz
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PIL import Image

# ============================================================
# 1. SETTINGS
# ============================================================

REGIONS = {
    # river canyon: the Nho Que river cuts a gorge more than 1000 m deep
    "nhoque": dict(
        name="Nho Quế river canyon, Hà Giang",
        lat=(23.12, 23.34), lon=(105.33, 105.54),
        a=(23.235, 105.37), b=(23.275, 105.49),      # (lat, lon) of A and B
        river=("Nho Quế", "Sông Nho Quế"),           # (text in the OSM name, label)
        contour=(100, 500)),                          # (step, thick step) in m
    # isolated mountain next to a wide river plain
    "bavi": dict(
        name="Ba Vì mountain, Hà Nội",
        lat=(20.95, 21.20), lon=(105.20, 105.55),
        a=(21.058, 105.22), b=(21.058, 105.52),
        river=("Sông Đà", "Sông Đà"),
        contour=(100, 500)),
}

REGION = sys.argv[1] if len(sys.argv) > 1 else "nhoque"
cfg = REGIONS[REGION]
LAT_MIN, LAT_MAX = cfg["lat"]
LON_MIN, LON_MAX = cfg["lon"]
PROFILE_A, PROFILE_B = cfg["a"], cfg["b"]
CONTOUR_STEP, INDEX_STEP = cfg["contour"]

DATA_DIR = "data"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
OVERPASS_URLS = [                        # public Overpass servers, tried in order
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
R_EARTH = 6371.0                         # km

sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese names in the console

# ============================================================
# 2. LOAD THE DEM
# ============================================================


def tile_name(lat, lon):
    """Name of the 1 x 1 degree tile whose south-west corner is (lat, lon)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def read_tile(lat, lon):
    """Elevation array of one tile (downloaded once, then read from DATA_DIR).
    Returns None where no tile exists (open sea)."""
    name = tile_name(lat, lon)
    path = os.path.join(DATA_DIR, name + ".tif")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {name} (about 50 MB) ...")
        try:
            urllib.request.urlretrieve(DEM_URL.format(name=name), path + ".part")
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            raise
        os.replace(path + ".part", path)
    return np.array(Image.open(path), dtype=float)


def load_dem(lat_min, lat_max, lon_min, lon_max):
    """Mosaic the tiles that cover the region and crop it.
    Returns lat (north -> south), lon (west -> east) and elevation z[lat, lon]."""
    tile_lats = range(math.floor(lat_min), math.ceil(lat_max))
    tile_lons = range(math.floor(lon_min), math.ceil(lon_max))

    rows = []
    for tlat in reversed(tile_lats):                 # north to south
        tiles = [read_tile(tlat, tlon) for tlon in tile_lons]
        shape = next((t.shape for t in tiles if t is not None), (3600, 3600))
        rows.append(np.hstack([t if t is not None else np.zeros(shape) for t in tiles]))
    z = np.vstack(rows)

    # pixel (0, 0) is the north-west corner of the mosaic
    dlat = len(tile_lats) / z.shape[0]
    dlon = len(tile_lons) / z.shape[1]
    lat_top, lon_left = max(tile_lats) + 1, min(tile_lons)
    i0 = round((lat_top - lat_max) / dlat)
    i1 = min(round((lat_top - lat_min) / dlat), z.shape[0] - 1)
    j0 = round((lon_min - lon_left) / dlon)
    j1 = min(round((lon_max - lon_left) / dlon), z.shape[1] - 1)

    lat = lat_top - dlat * np.arange(i0, i1 + 1)
    lon = lon_left + dlon * np.arange(j0, j1 + 1)
    return lat, lon, z[i0:i1 + 1, j0:j1 + 1]


lat, lon, z = load_dem(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
dlat = lat[0] - lat[1]
dlon = lon[1] - lon[0]
print(f"{cfg['name']}")
print(f"DEM: {z.shape[0]} x {z.shape[1]} points, elevation {z.min():.0f} .. {z.max():.0f} m")

# ============================================================
# 3. LOAD THE RIVERS (OpenStreetMap)
# ============================================================


def load_rivers():
    """River centre lines inside the region, cached in DATA_DIR.
    Returns a list of (name, lat array, lon array)."""
    path = os.path.join(DATA_DIR, f"rivers_{REGION}.json")
    if not os.path.exists(path):
        query = (f'[out:json][timeout:60];way["waterway"="river"]'
                 f'({LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX});out tags geom;')
        body = urllib.parse.urlencode({"data": query}).encode()
        print("downloading rivers from OpenStreetMap ...")
        text = None
        for attempt in range(2):
            for url in OVERPASS_URLS:
                request = urllib.request.Request(
                    url, data=body, headers={"User-Agent": "SPMM26-terrain-script/1.0"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        text = response.read().decode("utf-8")
                    break
                except (urllib.error.URLError, TimeoutError) as err:
                    print(f"  {url}: {err}")
            if text is not None:
                break
            time.sleep(10)                    # busy servers: wait and try again
        try:
            data = json.loads(text) if text else {}
        except ValueError:                    # an HTML error page instead of JSON
            data = {}
        if not data.get("elements"):
            print("  could not download rivers, drawing the map without them",
                  data.get("remark", ""))
            return []
        with open(path, "w", encoding="utf-8") as f:     # cache only good answers
            f.write(text)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    rivers = []
    for way in data.get("elements", []):
        points = way.get("geometry", [])
        rivers.append((way.get("tags", {}).get("name", ""),
                       np.array([p["lat"] for p in points]),
                       np.array([p["lon"] for p in points])))
    return rivers


rivers = load_rivers()
river_key, river_label = cfg["river"]
main_rivers = [r for r in rivers if river_key in r[0]]
print(f"rivers: {len(rivers)} OSM ways, {len(main_rivers)} of them '{river_label}'")

# ============================================================
# 4. LOCAL COORDINATES IN KILOMETRES
# ============================================================
# Over a few tens of km the Earth is flat enough for an equirectangular
# projection: x grows east, y grows north, both in km.

lat_c = math.radians((LAT_MIN + LAT_MAX) / 2)


def to_km(lat_deg, lon_deg):
    x = R_EARTH * math.cos(lat_c) * np.radians(np.asarray(lon_deg) - LON_MIN)
    y = R_EARTH * np.radians(np.asarray(lat_deg) - LAT_MIN)
    return x, y


x_km, y_km = to_km(lat, lon)
dx_m = 1000 * R_EARTH * math.cos(lat_c) * math.radians(dlon)    # grid spacing in m
dy_m = 1000 * R_EARTH * math.radians(dlat)
print(f"grid spacing: {dx_m:.1f} m (east-west) x {dy_m:.1f} m (north-south)")

# save for the population model (terrain2d.py style: x, y in km, z in m)
np.savez_compressed(os.path.join(DATA_DIR, f"dem_{REGION}.npz"),
                    lat=lat, lon=lon, x_km=x_km, y_km=y_km, z=z)

# ============================================================
# 5. CROSS-SECTION A-B
# ============================================================


def elevation_at(lat_pts, lon_pts):
    """Bilinear interpolation of the DEM."""
    fi = (lat[0] - lat_pts) / dlat           # fractional row index
    fj = (lon_pts - lon[0]) / dlon           # fractional column index
    i = np.clip(np.floor(fi).astype(int), 0, z.shape[0] - 2)
    j = np.clip(np.floor(fj).astype(int), 0, z.shape[1] - 2)
    a, b = fi - i, fj - j
    return ((1 - a) * (1 - b) * z[i, j] + (1 - a) * b * z[i, j + 1]
            + a * (1 - b) * z[i + 1, j] + a * b * z[i + 1, j + 1])


def river_crossings():
    """Where A-B crosses a river: list of (distance from A in km, river name)."""
    ax_, ay_ = to_km(*PROFILE_A)
    ex, ey = np.subtract(to_km(*PROFILE_B), (ax_, ay_))
    found = []
    for name, rlat, rlon in rivers:
        rx, ry = to_km(rlat, rlon)
        for k in range(len(rx) - 1):
            fx, fy = rx[k + 1] - rx[k], ry[k + 1] - ry[k]
            den = ex * fy - ey * fx
            if den == 0:
                continue
            gx, gy = rx[k] - ax_, ry[k] - ay_
            t = (gx * fy - gy * fx) / den     # position along A-B (0..1)
            u = (gx * ey - gy * ex) / den     # position along the river segment (0..1)
            if 0 <= t <= 1 and 0 <= u <= 1:
                found.append((t * math.hypot(ex, ey), name))
    return found


n_samples = 800
s = np.linspace(0, 1, n_samples)
prof_lat = PROFILE_A[0] + s * (PROFILE_B[0] - PROFILE_A[0])
prof_lon = PROFILE_A[1] + s * (PROFILE_B[1] - PROFILE_A[1])
prof_z = elevation_at(prof_lat, prof_lon)
px, py = to_km(prof_lat, prof_lon)
dist = np.hypot(px - px[0], py - py[0])               # km along A-B
crossings = river_crossings()
print(f"profile A-B: {dist[-1]:.1f} km, elevation {prof_z.min():.0f} .. {prof_z.max():.0f} m")
for d, name in crossings:
    print(f"  crosses {name or '(unnamed river)'} at {d:.2f} km, "
          f"{np.interp(d, dist, prof_z):.0f} m")

# ============================================================
# 6. PLOTS
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

# one hue, light (low) -> dark (high)
cmap = LinearSegmentedColormap.from_list(
    "elevation", ["#f6f1e4", "#dcc9a0", "#b8925c", "#8a5f36", "#5a3920"])
norm = Normalize(vmin=100 * math.floor(z.min() / 100), vmax=z.max())
ink = "#3b2a1e"
river_color = "#2B6CB0"
line_color = "#A02020"
halo = [pe.withStroke(linewidth=2.5, foreground="white")]     # readable text on the map

# ---------- Figure R1: topographic map ----------
fig, ax = plt.subplots(figsize=(7, 5.6))

shade = LightSource(azdeg=315, altdeg=45).shade(
    z, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=1.5, dx=dx_m, dy=dy_m)
ax.imshow(shade, extent=[lon[0] - dlon / 2, lon[-1] + dlon / 2,
                         lat[-1] - dlat / 2, lat[0] + dlat / 2])
ax.set_aspect(1 / math.cos(lat_c))           # 1 km east = 1 km north on paper

LON, LAT = np.meshgrid(lon, lat)
first = CONTOUR_STEP * math.ceil(z.min() / CONTOUR_STEP)
minor = [h for h in np.arange(first, z.max(), CONTOUR_STEP) if h % INDEX_STEP]
ax.contour(LON, LAT, z, levels=minor, colors=ink, linewidths=0.35, alpha=0.7)
cs = ax.contour(LON, LAT, z, levels=np.arange(INDEX_STEP, z.max(), INDEX_STEP),
                colors=ink, linewidths=0.9)
for txt in ax.clabel(cs, fmt="%d m", fontsize=7):
    txt.set_path_effects(halo)

# rivers: the main river thicker
for name, rlat, rlon in rivers:
    main = river_key in name
    ax.plot(rlon, rlat, color=river_color, linewidth=2.2 if main else 1.0,
            solid_capstyle="round", path_effects=[pe.withStroke(linewidth=3.5 if main else 2,
                                                                 foreground="white")])
if main_rivers:
    pts = sorted((la, lo) for _, rlat, rlon in main_rivers for la, lo in zip(rlat, rlon)
                 if LAT_MIN < la < LAT_MAX and LON_MIN < lo < LON_MAX)
    la, lo = pts[len(pts) // 4]
    ax.annotate(river_label, (lo, la), textcoords="offset points", xytext=(8, 0),
                va="center", fontsize=10, fontstyle="italic", color=river_color,
                path_effects=halo)

# highest point
k = np.unravel_index(np.argmax(z), z.shape)
ax.plot(lon[k[1]], lat[k[0]], "^", color="#222222", markersize=7)
ax.annotate(f"{z.max():.0f} m", (lon[k[1]], lat[k[0]]), textcoords="offset points",
            xytext=(6, 9), fontsize=9, color="#222222", path_effects=halo)

# cross-section line
ax.plot([PROFILE_A[1], PROFILE_B[1]], [PROFILE_A[0], PROFILE_B[0]],
        color=line_color, linewidth=2)
for label, (plat, plon) in [("A", PROFILE_A), ("B", PROFILE_B)]:
    ax.plot(plon, plat, "o", color=line_color, markersize=5)
    ax.annotate(label, (plon, plat), textcoords="offset points", xytext=(0, 7),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=line_color, path_effects=halo)

# scale bar (5 km) on a white box
bar_km = 5
bar_deg = math.degrees(bar_km / (R_EARTH * math.cos(lat_c)))
scale_bar = AnchoredSizeBar(ax.transData, bar_deg, f"{bar_km} km", loc="lower left",
                            pad=0.4, borderpad=0.6, sep=3, frameon=True,
                            size_vertical=0.004 * (LAT_MAX - LAT_MIN), color="#222222",
                            fontproperties={"size": 9})
scale_bar.patch.set(alpha=0.85, edgecolor="#cccccc")
ax.add_artist(scale_bar)

# legend
ax.plot([], [], color=river_color, linewidth=2, label="River")
ax.plot([], [], color=line_color, linewidth=2, label="Cross-section A-B")
ax.plot([], [], "^", color="#222222", label="Highest point")
ax.legend(loc="upper right", fontsize=8, framealpha=0.85, edgecolor="#cccccc")

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"Topographic map: {cfg['name']}\n"
             f"contours every {CONTOUR_STEP} m, thick every {INDEX_STEP} m",
             fontsize=11, color="#222222")
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
             label="Elevation (m)", shrink=0.85)
fig.text(0.01, 0.01, "Data: Copernicus DEM GLO-30 (ESA); rivers © OpenStreetMap contributors",
         fontsize=7, color="#777777")
fig.tight_layout()
fig.savefig(f"FigR1_topo_map_{REGION}.png", dpi=250)

# ---------- Figure R2: vertical cross-section ----------
fig, ax = plt.subplots(figsize=(7, 3))

base = 100 * math.floor((prof_z.min() - 100) / 100) if prof_z.min() > 200 else 0
top = prof_z.max() + 0.15 * (prof_z.max() - base)
ax.fill_between(dist, prof_z, base, color="#dcc9a0")
ax.plot(dist, prof_z, color="#5a3920", linewidth=1.5)

kp = np.argmax(prof_z)
ax.plot(dist[kp], prof_z[kp], "^", color="#222222", markersize=7)
ax.annotate(f"{prof_z[kp]:.0f} m", (dist[kp], prof_z[kp]), textcoords="offset points",
            xytext=(6, 2), fontsize=9, color="#222222")

for d, name in crossings:
    zr = np.interp(d, dist, prof_z)
    main = river_key in name
    ax.plot(d, zr, "v", color=river_color, markersize=9 if main else 6,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    if main:
        ax.annotate(f"{river_label}\n{zr:.0f} m", (d, zr), textcoords="offset points",
                    xytext=(10, 4), va="bottom", fontsize=9, fontstyle="italic",
                    color=river_color, path_effects=halo)

ax.set_xlim(0, dist[-1])
ax.set_ylim(base, top)
for label, xpos, ha in [("A", 0, "left"), ("B", dist[-1], "right")]:
    ax.text(xpos, top, f" {label} ", ha=ha, va="top", fontsize=11,
            fontweight="bold", color=line_color)

ax.set_xlabel("Distance along A-B (km)")
ax.set_ylabel("Elevation (m)")
ax.grid(color="#e6e6e6", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

# vertical exaggeration = (horizontal m per pixel) / (vertical m per pixel)
bbox = ax.get_window_extent()
ve = (1000 * dist[-1] / bbox.width) / ((top - base) / bbox.height)
ax.set_title(f"Cross-section A-B  (vertical exaggeration about {ve:.0f}x)",
             fontsize=11, color="#222222")
fig.tight_layout()
fig.savefig(f"FigR2_profile_{REGION}.png", dpi=250)

plt.show()
