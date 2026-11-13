#!/usr/bin/env python3
"""
Track 1h precipitation clusters from pljusak.com and detect which ones are
heading toward a given target location.

Data source: the `var podaci = [...]` array embedded in the pljusak.com HTML.
Column mapping (see prikazi() in the site's JS):
  0=type  1=name  2=lat  3=lon  4=elev  5=zIndex  6=link
  7=hw    8=sw    9=shelter  10=time  11=webcam
  12=temp 13=temp_trend 14=pressure 15=pressure_trend 16=humidity
  17=wind_dir 18=wind_ms
  19=precip_month 20=precip_today 21=precip_24h 22=precip_1h
  23=precip_6h 24=max_temp 25=min_temp 26=max_wind 27=uv
  28=temp_24h_change 29=slug

Run:  python pljusak_track.py --lat 45.8146 --lon 15.9722 --poll 180
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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

DEFAULT_TARGET = (45.8146, 15.9722)      # Zagreb Grič
DEFAULT_POLL = 180                        # seconds between snapshots
MIN_PRECIP_MM = 0.1                       # station needs > this in 1h to count
CLUSTER_EPS_KM = 20.0                     # stations this close get clustered
CLUSTER_MIN_POINTS = 2                    # ignore single-station "clusters"
MAX_TRACK_DISTANCE_KM = 40.0              # max centroid jump between snapshots
THREAT_RADIUS_KM = 30.0                   # flag if closest-approach is within
MIN_SPEED_KMH = 2.0                       # below this, treat as stationary


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
    total_precip: float         # sum(mm/h) across stations in cluster
    max_precip: float           # strongest station in cluster
    n: int
    station_names: list[str]
    # Filled in by track_motion():
    velocity_kmh: tuple = (0.0, 0.0)   # (vx_east, vy_north)
    speed_kmh: float = 0.0


@dataclass
class Approach:
    dist_km: float
    bearing_to_target: float
    bearing_of_motion: Optional[float]
    approach_angle: Optional[float]   # 0 = straight at you, 90 = broadside
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
    """Return Station objects for every non-camera row with > min_mm in last 1h."""
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
# Spatial clustering (union-find, O(n^2) which is fine for ~100 wet stations)
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
    """Cluster wet stations; centroid weighted by precipitation intensity."""
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
            station_names=sorted(
                (s.name for s in group),
                key=lambda _: 0),  # keep insertion order
        ))
    return clusters


# ----------------------------------------------------------------------------
# Motion tracking between two snapshots
# ----------------------------------------------------------------------------
def track_motion(prev_clusters, new_clusters, dt_hours):
    """Match each new cluster to the nearest previous centroid within range."""
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
        # Displacement in a local east/north km plane
        lat_mid = (nc.centroid_lat + pc.centroid_lat) / 2
        dy_km = (nc.centroid_lat - pc.centroid_lat) * 111.32
        dx_km = (nc.centroid_lon - pc.centroid_lon) * 111.32 * math.cos(math.radians(lat_mid))
        vx = dx_km / dt_hours
        vy = dy_km / dt_hours
        nc.velocity_kmh = (vx, vy)
        nc.speed_kmh = math.hypot(vx, vy)


# ----------------------------------------------------------------------------
# Approach prediction (closest-approach distance + ETA)
# ----------------------------------------------------------------------------
def assess_approach(cluster, target_lat, target_lon):
    coslat = math.cos(math.radians((target_lat + cluster.centroid_lat) / 2))
    # Vector from target → cluster, in local east/north km
    dx_km = (cluster.centroid_lon - target_lon) * 111.32 * coslat
    dy_km = (cluster.centroid_lat - target_lat) * 111.32
    dist_km = math.hypot(dx_km, dy_km)

    # Compass bearing from cluster toward target (0=N, 90=E)
    bearing_to_target = math.degrees(math.atan2(-dx_km, -dy_km)) % 360.0

    vx, vy = cluster.velocity_kmh
    speed = math.hypot(vx, vy)

    if speed < MIN_SPEED_KMH:
        return Approach(dist_km, bearing_to_target, None, None, None, None, False)

    bearing_of_motion = math.degrees(math.atan2(vx, vy)) % 360.0

    # Unit vector from cluster → target
    ux, uy = -dx_km / dist_km, -dy_km / dist_km
    dot = vx * ux + vy * uy
    approach_angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / speed))))

    # Closest point of approach: t* = -(R·V)/|V|²
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
# Main loop
# ----------------------------------------------------------------------------
def run(target_lat, target_lon, poll_s, iterations=None, dump_jsonl=None):
    prev_clusters = []
    prev_time = None
    it = 0
    out_fh = open(dump_jsonl, "a") if dump_jsonl else None
    print(f"Target: ({target_lat:.4f}, {target_lon:.4f})")
    print(f"Poll: {poll_s}s | cluster eps: {CLUSTER_EPS_KM} km | "
          f"threat radius: {THREAT_RADIUS_KM} km\n")

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
    p.add_argument("--poll", type=int, default=DEFAULT_POLL,
                   help="seconds between snapshots (default 180)")
    p.add_argument("--iterations", type=int, default=None,
                   help="stop after N snapshots (default: run forever)")
    p.add_argument("--dump-jsonl", type=str, default=None,
                   help="append cluster snapshots to this JSONL file")
    p.add_argument("--min-precip", type=float, default=None,
                   help="override MIN_PRECIP_MM")
    p.add_argument("--eps-km", type=float, default=None,
                   help="override CLUSTER_EPS_KM")
    args = p.parse_args()

    global MIN_PRECIP_MM, CLUSTER_EPS_KM
    if args.min_precip is not None:
        MIN_PRECIP_MM = args.min_precip
    if args.eps_km is not None:
        CLUSTER_EPS_KM = args.eps_km

    run(args.lat, args.lon, args.poll, args.iterations, args.dump_jsonl)


if __name__ == "__main__":
    main()