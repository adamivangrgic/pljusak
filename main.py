#!/usr/bin/env python3
"""
Track 1h precipitation clusters from pljusak.com and detect which ones are
heading toward a given target location.  Renders an ASCII map with well-known
reference cities so the geography is readable.

Run:  python pljusak_track.py --lat 45.8146 --lon 15.9722 --poll 180
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SOURCE_URL = "https://pljusak.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; pljusak-precip-tracker/1.0)",
    "Referer": "https://pljusak.com/",
    "Accept": "text/html,application/xhtml+xml",
}

DEFAULT_TARGET = (45.8146, 15.9722)
DEFAULT_POLL = 180
MIN_PRECIP_MM = 0.1
CLUSTER_EPS_KM = 20.0
CLUSTER_MIN_POINTS = 2
MAX_TRACK_DISTANCE_KM = 40.0
THREAT_RADIUS_KM = 30.0
MIN_SPEED_KMH = 2.0

# ASCII rendering
ASCII_WIDTH = 84
ASCII_HEIGHT = 32
ASCII_PAD_DEG = 1.2

# Reference cities drawn on the map.  Keep labels short (2–3 chars) so
# they fit in the map without clobbering each other.
# (label, lat, lon)
CITIES = [
    ("zg", 45.8146, 15.9722),   # Zagreb
    ("ri", 45.3271, 14.4422),   # Rijeka
    ("st", 43.5081, 16.4402),   # Split
    ("os", 45.5550, 18.6955),   # Osijek
    ("zd", 44.1194, 15.2314),   # Zadar
    ("db", 42.6507, 18.0944),   # Dubrovnik
    ("pu", 44.8666, 13.8496),   # Pula
    ("vk", 45.2551, 19.1122),   # Vukovar
    ("ka", 45.4929, 15.5551),   # Karlovac
    ("va", 46.3057, 16.3366),   # Varaždin
    ("bk", 45.3271, 14.4422),   # (placeholder - remove if not needed)
    ("lj", 46.0569, 14.5058),   # Ljubljana (SLO)
    ("tb", 43.8633, 18.4131),   # Sarajevo (BiH)
]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_label(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) % 360 // 22.5)]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Station:
    lat: float
    lon: float
    precip_1h: float
    name: str
    kind: str


@dataclass
class Cluster:
    centroid_lat: float
    centroid_lon: float
    total_precip: float
    max_precip: float
    n: int
    station_names: list[str] = field(default_factory=list)
    members: list = field(default_factory=list)
    velocity_kmh: tuple = (0.0, 0.0)
    speed_kmh: float = 0.0


@dataclass
class Approach:
    dist_km: float
    bearing_to_target: float
    bearing_of_motion: Optional[float]
    approach_angle: Optional[float]
    closest_approach_km: Optional[float]
    eta_min: Optional[float]
    approaching: bool


# ----------------------------------------------------------------------------
# Fetch + extract
# ----------------------------------------------------------------------------
def fetch_podaci(timeout=30):
    r = requests.get(SOURCE_URL, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    m = re.search(r"var\s+podaci\s*=\s*(\[.*?\]);\s*</script>", r.text, re.S)
    if not m:
        raise RuntimeError("Could not locate `var podaci = [...]` in HTML")
    return json.loads(m.group(1))


def extract_wet_stations(rows, min_mm=MIN_PRECIP_MM):
    out = []
    for row in rows:
        if not row or row[0] == "kamera":
            continue
        try:
            lat = float(row[2])
            lon = float(row[3])
            v = float(row[22])
        except (TypeError, ValueError, IndexError):
            continue
        if v <= min_mm:
            continue
        out.append(Station(lat=lat, lon=lon, precip_1h=v,
                           name=row[1], kind=row[0]))
    return out


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
def _cluster_by_distance(points, eps_km):
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if haversine_km(pi.lat, pi.lon, pj.lat, pj.lon) <= eps_km:
                union(i, j)

    groups = defaultdict(list)
    for i, p in enumerate(points):
        groups[find(i)].append(p)
    return list(groups.values())


def build_clusters(stations, eps_km=CLUSTER_EPS_KM,
                   min_pts=CLUSTER_MIN_POINTS):
    clusters = []
    for group in _cluster_by_distance(stations, eps_km):
        if len(group) < min_pts:
            continue
        wsum = sum(s.precip_1h for s in group)
        if wsum <= 0:
            continue
        clat = sum(s.lat * s.precip_1h for s in group) / wsum
        clon = sum(s.lon * s.precip_1h for s in group) / wsum
        clusters.append(Cluster(
            centroid_lat=clat, centroid_lon=clon,
            total_precip=wsum,
            max_precip=max(s.precip_1h for s in group),
            n=len(group),
            station_names=[s.name for s in group],
            members=list(group),
        ))
    return clusters


# ----------------------------------------------------------------------------
# Motion & approach
# ----------------------------------------------------------------------------
def track_motion(prev_clusters, new_clusters, dt_hours):
    if not prev_clusters or not new_clusters or dt_hours <= 0:
        return
    used = set()
    for nc in new_clusters:
        best_i, best_d = None, MAX_TRACK_DISTANCE_KM
        for i, pc in enumerate(prev_clusters):
            if i in used:
                continue
            d = haversine_km(nc.centroid_lat, nc.centroid_lon,
                             pc.centroid_lat, pc.centroid_lon)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            continue
        used.add(best_i)
        pc = prev_clusters[best_i]
        lat_mid = (nc.centroid_lat + pc.centroid_lat) / 2
        dy_km = (nc.centroid_lat - pc.centroid_lat) * 111.32
        dx_km = (nc.centroid_lon - pc.centroid_lon) * 111.32 * math.cos(math.radians(lat_mid))
        vx = dx_km / dt_hours
        vy = dy_km / dt_hours
        nc.velocity_kmh = (vx, vy)
        nc.speed_kmh = math.hypot(vx, vy)


def assess_approach(cluster, target_lat, target_lon):
    coslat = math.cos(math.radians((target_lat + cluster.centroid_lat) / 2))
    dx_km = (cluster.centroid_lon - target_lon) * 111.32 * coslat
    dy_km = (cluster.centroid_lat - target_lat) * 111.32
    dist_km = math.hypot(dx_km, dy_km)

    bearing_to_target = math.degrees(math.atan2(-dx_km, -dy_km)) % 360.0

    vx, vy = cluster.velocity_kmh
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH:
        return Approach(dist_km, bearing_to_target, None, None, None, None, False)

    bearing_of_motion = math.degrees(math.atan2(vx, vy)) % 360.0
    ux, uy = -dx_km / dist_km, -dy_km / dist_km
    dot = vx * ux + vy * uy
    approach_angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / speed))))

    t_star = -(dx_km * vx + dy_km * vy) / (speed * speed)
    approaching = t_star > 0
    if approaching:
        cax = dx_km + vx * t_star
        cay = dy_km + vy * t_star
        closest_km = math.hypot(cax, cay)
        eta_min = t_star * 60.0
    else:
        closest_km = dist_km
        eta_min = None

    return Approach(dist_km, bearing_to_target, bearing_of_motion,
                    approach_angle, closest_km, eta_min, approaching)


# ----------------------------------------------------------------------------
# ASCII rendering
# ----------------------------------------------------------------------------
def render_ascii(stations, clusters, target_lat, target_lon,
                 width=ASCII_WIDTH, height=ASCII_HEIGHT,
                 pad_deg=ASCII_PAD_DEG, show_cities=True):
    """
    Draw a lat/lon grid as text with reference cities.

    Marker priority (higher wins):
        T  target
        C  cluster centroid
        lowercase city label letters
        motion arrows
        · * #  station markers (by precip intensity)
    """
    # --- Bounding box -------------------------------------------------------
    lats = [s.lat for s in stations] + [target_lat]
    lons = [s.lon for s in stations] + [target_lon]
    if show_cities:
        lats += [c[1] for c in CITIES]
        lons += [c[2] for c in CITIES]
    lat_min, lat_max = min(lats) - pad_deg, max(lats) + pad_deg
    lon_min, lon_max = min(lons) - pad_deg, max(lons) + pad_deg

    mid_lat = (lat_min + lat_max) / 2.0
    cos_lat = max(0.05, math.cos(math.radians(mid_lat)))
    width_adj = max(20, int(width * cos_lat))
    height_adj = max(8, height)

    lon_span = max(1e-6, lon_max - lon_min)
    lat_span = max(1e-6, lat_max - lat_min)

    def to_grid(lat, lon):
        col = int(round((lon - lon_min) / lon_span * (width_adj - 1)))
        row = int(round((lat_max - lat) / lat_span * (height_adj - 1)))
        return max(0, min(height_adj - 1, row)), max(0, min(width_adj - 1, col))

    canvas = [[" "] * width_adj for _ in range(height_adj)]

    # --- Pass 1: station dots (lowest priority of everything) ---------------
    for s in stations:
        r, c = to_grid(s.lat, s.lon)
        if s.precip_1h >= 10:
            ch = "#"
        elif s.precip_1h >= 3:
            ch = "*"
        else:
            ch = "·"
        if canvas[r][c] == " ":
            canvas[r][c] = ch

    # --- Pass 2: motion arrows ---------------------------------------------
    for cl in clusters:
        if cl.speed_kmh < MIN_SPEED_KMH:
            continue
        vx, vy = cl.velocity_kmh
        bearing = math.degrees(math.atan2(vx, vy)) % 360.0
        if bearing < 22.5 or bearing >= 337.5:
            arrow = "↑"
        elif bearing < 67.5:
            arrow = "↗"
        elif bearing < 112.5:
            arrow = "→"
        elif bearing < 157.5:
            arrow = "↘"
        elif bearing < 202.5:
            arrow = "↓"
        elif bearing < 247.5:
            arrow = "↙"
        elif bearing < 292.5:
            arrow = "←"
        else:
            arrow = "↖"
        r, c = to_grid(cl.centroid_lat, cl.centroid_lon)
        ar = r - int(round(2 * math.cos(math.radians(bearing))))
        ac = c + int(round(2 * math.sin(math.radians(bearing))))
        if 0 <= ar < height_adj and 0 <= ac < width_adj:
            if canvas[ar][ac] in (" ", "·", "*", "#"):
                canvas[ar][ac] = arrow

    # --- Pass 3: city labels -----------------------------------------------
    # Draw each city as its 2-char label.  Reserve a 1-cell halo so
    # subsequent markers (C, T) can still overwrite, but stations can't
    # bleed into the label.
    if show_cities:
        reserved = set()
        for label, clat, clon in CITIES:
            r, c = to_grid(clat, clon)
            for i, ch in enumerate(label):
                cc = c + i
                if 0 <= cc < width_adj and canvas[r][cc] not in ("T", "C"):
                    canvas[r][cc] = ch
                reserved.add((r, cc))
            # halo around label so station dots don't crowd it
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1, len(label)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height_adj and 0 <= cc < width_adj:
                        reserved.add((rr, cc))
        # Knock out any station dots sitting directly on the halo so the
        # label reads cleanly.  (We re-draw stations first, so this is a
        # targeted cleanup.)
        for (rr, cc) in reserved:
            if canvas[rr][cc] in ("·", "*", "#"):
                canvas[rr][cc] = " "

    # --- Pass 4: cluster centroids -----------------------------------------
    for cl in clusters:
        r, c = to_grid(cl.centroid_lat, cl.centroid_lon)
        canvas[r][c] = "C"

    # --- Pass 5: target ----------------------------------------------------
    tr, tc = to_grid(target_lat, target_lon)
    # Wipe any label letters that overlap the target cell so T is unambiguous
    canvas[tr][tc] = "T"

    # --- Frame -------------------------------------------------------------
    top = "┌" + "─" * width_adj + "┐"
    bot = "└" + "─" * width_adj + "┘"
    lines = [top]
    for row in canvas:
        lines.append("│" + "".join(row) + "│")
    lines.append(bot)

    legend = (f"  T=target  C=cluster  →=motion  "
              f"·/*/#=1h precip (<3/3–10/≥10 mm)  "
              f"lowercase=reference city  "
              f"bbox {lat_min:.2f}–{lat_max:.2f}N, {lon_min:.2f}–{lon_max:.2f}E")
    lines.append(legend)
    if show_cities:
        city_line = "  cities: " + "  ".join(f"{lbl}={clat:.2f},{clon:.2f}"
                                             for lbl, clat, clon in CITIES[:8])
        lines.append(city_line)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(target_lat, target_lon, poll_s, iterations=None, dump_jsonl=None,
        show_ascii=True, show_cities=True):
    prev_clusters = []
    prev_time = None
    it = 0
    out_fh = open(dump_jsonl, "a") if dump_jsonl else None
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Poll: {poll_s}s | cluster eps: {CLUSTER_EPS_KM} km | "
          f"threat radius: {THREAT_RADIUS_KM} km")
    print(f"ASCII: {'on' if show_ascii else 'off'} | "
          f"cities: {'on' if show_cities else 'off'}\n")

    try:
        while True:
            it += 1
            if iterations is not None and it > iterations:
                return
            try:
                rows = fetch_podaci()
                stations = extract_wet_stations(rows)
                now = datetime.now(timezone.utc)
                clusters = build_clusters(stations)
                if prev_time is not None:
                    dt_h = (now - prev_time).total_seconds() / 3600.0
                    track_motion(prev_clusters, clusters, dt_h)

                print(f"\n=== {now.strftime('%H:%M:%SZ')} | "
                      f"wet stations: {len(stations)} | clusters: {len(clusters)} ===")

                if show_ascii and stations:
                    print(render_ascii(stations, clusters,
                                       target_lat, target_lon,
                                       show_cities=show_cities))

                threats = []
                for c in clusters:
                    a = assess_approach(c, target_lat, target_lon)
                    line = (f"  [{c.n:2d} st | Σ {c.total_precip:5.1f} mm/h | "
                            f"max {c.max_precip:4.1f}] @({c.centroid_lat:.3f},{c.centroid_lon:.3f})"
                            f"  d={a.dist_km:5.1f}km")
                    if a.bearing_of_motion is not None:
                        line += f"  v={c.speed_kmh:4.1f}km/h→{bearing_label(a.bearing_of_motion)}"
                        line += f"  ang={a.approach_angle:3.0f}°"
                    if a.approaching and a.eta_min is not None:
                        line += f"  ETA={a.eta_min:4.0f}min  closest={a.closest_approach_km:4.1f}km"
                    print(line)

                    if (a.approaching and a.closest_approach_km is not None
                            and a.closest_approach_km <= THREAT_RADIUS_KM
                            and a.approach_angle is not None
                            and a.approach_angle <= 60):
                        threats.append((c, a))

                    if out_fh:
                        out_fh.write(json.dumps({
                            "ts": now.isoformat(),
                            "lat": c.centroid_lat, "lon": c.centroid_lon,
                            "n": c.n, "total": c.total_precip, "max": c.max_precip,
                            "speed_kmh": c.speed_kmh,
                            "bearing_deg": a.bearing_of_motion,
                            "dist_km": a.dist_km,
                            "approach_angle": a.approach_angle,
                            "closest_km": a.closest_approach_km,
                            "eta_min": a.eta_min,
                            "approaching": a.approaching,
                            "stations": c.station_names,
                        }) + "\n")

                if threats:
                    print(f"\n  ⚠ {len(threats)} cluster(s) approaching within "
                          f"{THREAT_RADIUS_KM:.0f} km:")
                    for c, a in threats:
                        print(f"     • {c.station_names[0]}… "
                              f"ETA {a.eta_min:.0f} min, "
                              f"closest {a.closest_approach_km:.1f} km, "
                              f"peak {c.max_precip:.1f} mm/h")

                prev_clusters = clusters
                prev_time = now

            except Exception as e:
                print(f"  fetch error: {e}", file=sys.stderr)

            try:
                time.sleep(poll_s)
            except KeyboardInterrupt:
                return
    finally:
        if out_fh:
            out_fh.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=DEFAULT_TARGET[0])
    p.add_argument("--lon", type=float, default=DEFAULT_TARGET[1])
    p.add_argument("--poll", type=int, default=DEFAULT_POLL)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--dump-jsonl", type=str, default=None)
    p.add_argument("--min-precip", type=float, default=None)
    p.add_argument("--eps-km", type=float, default=None)
    p.add_argument("--ascii", dest="ascii", action="store_true", default=True)
    p.add_argument("--no-ascii", dest="ascii", action="store_false")
    p.add_argument("--cities", dest="cities", action="store_true", default=True)
    p.add_argument("--no-cities", dest="cities", action="store_false")
    args = p.parse_args()

    global MIN_PRECIP_MM, CLUSTER_EPS_KM
    if args.min_precip is not None:
        MIN_PRECIP_MM = args.min_precip
    if args.eps_km is not None:
        CLUSTER_EPS_KM = args.eps_km

    run(args.lat, args.lon, args.poll, args.iterations, args.dump_jsonl,
        args.ascii, args.cities)


if __name__ == "__main__":
    main()
