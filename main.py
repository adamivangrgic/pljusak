#!/usr/bin/env python3
"""
Track 1h precipitation clusters from pljusak.com and detect which ones are
heading toward a given target location.  The target can be specified by
coordinates, city name/label, or by substring of a station name.

Run examples:
  python pljusak_track.py --location zg
  python pljusak_track.py --location "Split"
  python pljusak_track.py --location Puntijarka
  python pljusak_track.py --lat 45.8146 --lon 15.9722
"""

import argparse
import difflib
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

ASCII_WIDTH = 84
ASCII_HEIGHT = 32
MAP_RADIUS_KM = 100.0

# Reference cities: (label, name, lat, lon)
CITIES = [
    ("zg", "Zagreb",     45.8146, 15.9722),
    ("ri", "Rijeka",     45.3271, 14.4422),
    ("st", "Split",      43.5081, 16.4402),
    ("os", "Osijek",     45.5550, 18.6955),
    ("zd", "Zadar",      44.1194, 15.2314),
    ("db", "Dubrovnik",  42.6507, 18.0944),
    ("pu", "Pula",       44.8666, 13.8496),
    ("vk", "Vukovar",    45.2551, 19.1122),
    ("ka", "Karlovac",   45.4929, 15.5551),
    ("va", "Varaždin",   46.3057, 16.3366),
    ("lj", "Ljubljana",  46.0569, 14.5058),
    ("tb", "Sarajevo",   43.8633, 18.4131),
    ("sk", "Sisak",      45.4667, 16.3789),
    ("vk2", "Vinkovci",  45.2886, 18.8042),
    ("kt", "Kutina",     45.4772, 16.7783),
    ("bj", "Bjelovar",   45.8986, 16.8422),
    ("po", "Požega",     45.3403, 17.6853),
    ("sb", "Slavonski Brod", 45.1603, 18.0156),
    ("mk", "Makarska",   43.2969, 17.0178),
    ("šb", "Šibenik",    43.7350, 15.8956),
    ("ri2","Rovinj",     45.0811, 13.6387),
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


def radius_to_bbox(lat, lon, radius_km):
    dlat = radius_km / 111.32
    coslat = max(0.01, math.cos(math.radians(lat)))
    dlon = radius_km / (111.32 * coslat)
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


# ----------------------------------------------------------------------------
# Location resolution
# ----------------------------------------------------------------------------
def resolve_city(query: str):
    """Match a city label (zg) or name (Zagreb) — case insensitive, fuzzy."""
    q = query.strip().lower()
    for label, name, lat, lon in CITIES:
        if q == label.lower() or q == name.lower():
            return name, lat, lon
    # fuzzy on both label and name
    by_label = {lbl.lower(): (nm, la, lo) for lbl, nm, la, lo in CITIES}
    by_name = {nm.lower(): (nm, la, lo) for lbl, nm, la, lo in CITIES}
    for source in (by_label, by_name):
        m = difflib.get_close_matches(q, list(source.keys()), n=1, cutoff=0.6)
        if m:
            return source[m[0]]
    return None


def resolve_station(query: str, rows):
    """Find stations whose name contains query (case insensitive, accent-tolerant)."""
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    q = norm(query)
    if not q:
        return None
    exact = []
    partial = []
    for row in rows:
        if not row or row[0] == "kamera":
            continue
        name = row[1] if len(row) > 1 else ""
        try:
            lat = float(row[2]); lon = float(row[3])
        except (TypeError, ValueError, IndexError):
            continue
        if norm(name) == q:
            exact.append((name, lat, lon))
        elif q in norm(name):
            partial.append((name, lat, lon))
    return (exact or partial or None)


def resolve_location(args, rows=None):
    """Turn args into (name, lat, lon).  Priority: --location > --lat/--lon > default."""
    if args.location:
        city = resolve_city(args.location)
        if city:
            return city
        # fall back to station lookup if we have data
        if rows is not None:
            matches = resolve_station(args.location, rows)
            if matches:
                if len(matches) > 1:
                    print(f"Multiple stations match '{args.location}': "
                          f"{[m[0] for m in matches[:5]]}... using first",
                          file=sys.stderr)
                return matches[0]
        print(f"ERROR: could not resolve location '{args.location}'",
              file=sys.stderr)
        sys.exit(2)
    return ("custom", args.lat, args.lon)


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
            lat = float(row[2]); lon = float(row[3]); v = float(row[22])
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
        nc.velocity_kmh = (dx_km / dt_hours, dy_km / dt_hours)
        nc.speed_kmh = math.hypot(*nc.velocity_kmh)


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
# ASCII rendering (unchanged from previous version except CITIES shape)
# ----------------------------------------------------------------------------
def render_ascii(stations, clusters, target_lat, target_lon,
                 radius_km=MAP_RADIUS_KM,
                 width=ASCII_WIDTH, height=ASCII_HEIGHT,
                 show_cities=True):
    lat_min, lat_max, lon_min, lon_max = radius_to_bbox(target_lat, target_lon, radius_km)
    mid_lat = (lat_min + lat_max) / 2.0
    cos_lat = max(0.05, math.cos(math.radians(mid_lat)))
    width_adj = max(20, int(width * cos_lat))
    height_adj = max(8, height)
    lon_span = max(1e-6, lon_max - lon_min)
    lat_span = max(1e-6, lat_max - lat_min)

    def in_box(lat, lon):
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    def to_grid(lat, lon):
        col = int(round((lon - lon_min) / lon_span * (width_adj - 1)))
        row = int(round((lat_max - lat) / lat_span * (height_adj - 1)))
        return max(0, min(height_adj - 1, row)), max(0, min(width_adj - 1, col))

    canvas = [[" "] * width_adj for _ in range(height_adj)]

    for s in stations:
        if not in_box(s.lat, s.lon):
            continue
        r, c = to_grid(s.lat, s.lon)
        ch = "#" if s.precip_1h >= 10 else "*" if s.precip_1h >= 3 else "·"
        if canvas[r][c] == " ":
            canvas[r][c] = ch

    for cl in clusters:
        if cl.speed_kmh < MIN_SPEED_KMH or not in_box(cl.centroid_lat, cl.centroid_lon):
            continue
        bearing = math.degrees(math.atan2(cl.velocity_kmh[0], cl.velocity_kmh[1])) % 360
        arrow = ("↑" if bearing < 22.5 or bearing >= 337.5 else
                 "↗" if bearing < 67.5 else "→" if bearing < 112.5 else
                 "↘" if bearing < 157.5 else "↓" if bearing < 202.5 else
                 "↙" if bearing < 247.5 else "←" if bearing < 292.5 else "↖")
        r, c = to_grid(cl.centroid_lat, cl.centroid_lon)
        ar = r - int(round(2 * math.cos(math.radians(bearing))))
        ac = c + int(round(2 * math.sin(math.radians(bearing))))
        if 0 <= ar < height_adj and 0 <= ac < width_adj and canvas[ar][ac] in (" ", "·", "*", "#"):
            canvas[ar][ac] = arrow

    if show_cities:
        reserved = set()
        for label, name, clat, clon in CITIES:
            if not in_box(clat, clon):
                continue
            r, c = to_grid(clat, clon)
            for i, ch in enumerate(label):
                cc = c + i
                if 0 <= cc < width_adj and canvas[r][cc] not in ("T", "C"):
                    canvas[r][cc] = ch
                reserved.add((r, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1, len(label)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height_adj and 0 <= cc < width_adj:
                        reserved.add((rr, cc))
        for (rr, cc) in reserved:
            if canvas[rr][cc] in ("·", "*", "#"):
                canvas[rr][cc] = " "

    for cl in clusters:
        if not in_box(cl.centroid_lat, cl.centroid_lon):
            continue
        r, c = to_grid(cl.centroid_lat, cl.centroid_lon)
        canvas[r][c] = "C"

    tr, tc = to_grid(target_lat, target_lon)
    canvas[tr][tc] = "T"

    top = "┌" + "─" * width_adj + "┐"
    bot = "└" + "─" * width_adj + "┘"
    lines = [top]
    for row in canvas:
        lines.append("│" + "".join(row) + "│")
    lines.append(bot)
    lines.append(f"  T=target  C=cluster  →=motion  ·/*/#=1h precip  "
                 f"lowercase=city  ±{radius_km:.0f}km")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(target_name, target_lat, target_lon, poll_s, iterations=None,
        dump_jsonl=None, show_ascii=True, show_cities=True,
        radius_km=MAP_RADIUS_KM):
    prev_clusters = []
    prev_time = None
    it = 0
    out_fh = open(dump_jsonl, "a") if dump_jsonl else None
    print(f"Target: {target_name} ({target_lat:.4f}, {target_lon:.4f})  "
          f"radius: {radius_km:.0f} km")
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
                                       radius_km=radius_km,
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
    p.add_argument("--location", type=str, default=None,
                   help="city label (zg), city name (Zagreb), or station name substring")
    p.add_argument("--lat", type=float, default=DEFAULT_TARGET[0])
    p.add_argument("--lon", type=float, default=DEFAULT_TARGET[1])
    p.add_argument("--poll", type=int, default=DEFAULT_POLL)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--dump-jsonl", type=str, default=None)
    p.add_argument("--min-precip", type=float, default=None)
    p.add_argument("--eps-km", type=float, default=None)
    p.add_argument("--radius", type=float, default=MAP_RADIUS_KM)
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

    # Resolve location.  Station lookup needs one fetch; cache it for the
    # first iteration so we don't pay for two GETs at startup.
    target_name, target_lat, target_lon = (None, args.lat, args.lon)
    pre_rows = None
    if args.location:
        city = resolve_city(args.location)
        if city:
            target_name, target_lat, target_lon = city
        else:
            try:
                pre_rows = fetch_podaci()
            except Exception as e:
                print(f"ERROR: could not fetch podaci to resolve station: {e}",
                      file=sys.stderr)
                sys.exit(2)
            matches = resolve_station(args.location, pre_rows)
            if not matches:
                print(f"ERROR: no city or station matches '{args.location}'",
                      file=sys.stderr)
                sys.exit(2)
            if len(matches) > 1:
                print(f"Multiple stations match '{args.location}': "
                      f"{[m[0] for m in matches[:5]]}; using '{matches[0][0]}'",
                      file=sys.stderr)
            target_name, target_lat, target_lon = matches[0]

    if not target_name:
        target_name = f"{target_lat:.4f},{target_lon:.4f}"

    run(target_name, target_lat, target_lon, args.poll, args.iterations,
        args.dump_jsonl, args.ascii, args.cities, args.radius)


if __name__ == "__main__":
    main()
