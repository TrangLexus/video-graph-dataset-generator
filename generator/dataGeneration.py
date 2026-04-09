#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
genev8_partition_dirs.py — Spatio-Temporal KG CSV Generator

Giữ nguyên output gốc ở thư mục --out:
- nodes_location.csv
- nodes_camera.csv
- partitions.csv
- nodes_person.csv
- nodes_thing.csv
- nodes_vehicle.csv
- nodes_timewindow.csv
- nodes_video.csv
- nodes_person_TW.csv
- nodes_thing_TW.csv
- nodes_vehicle_TW.csv
- rels.csv

Đồng thời tách thêm dữ liệu theo partition_id vào:
- by_partition/partition_0001/
- by_partition/partition_0002/
- ...

Trong mỗi thư mục partition sẽ có:
- nodes_location.csv
- nodes_camera.csv
- partitions.csv
- nodes_timewindow.csv
- nodes_video.csv
- nodes_person_TW.csv
- nodes_thing_TW.csv
- nodes_vehicle_TW.csv
- rels.csv

Các bảng global như nodes_person.csv / nodes_thing.csv / nodes_vehicle.csv
vẫn giữ ở thư mục gốc để tránh phá downstream hiện có.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, time
import json
from typing import Dict, List, Tuple
from collections import defaultdict


TW_SECONDS_FIXED = 10
INDOOR_LOC_TYPES = {"Indoor", "Office"}
OUTDOOR_LOC_TYPES = {"Outdoor", "Road", "Parking", "Garage"}
ALLOWED_LOC_TYPES = tuple(sorted(INDOOR_LOC_TYPES | OUTDOOR_LOC_TYPES))


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def hhmmss(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def yyyy_mm_dd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def sample_interval_within_tw(
    rng: random.Random,
    tw_start: datetime,
    tw_end: datetime,
    min_seconds: int = 2,
    max_seconds: int = 8,
) -> Tuple[datetime, datetime]:
    """
    Pick a random sub-interval within a time window.

    The returned (start, end) tuple satisfies tw_start <= start < end <= tw_end,
    with duration between min_seconds and max_seconds (clamped to fit within the window).
    """
    min_seconds = max(1, int(min_seconds))
    max_seconds = max(min_seconds, int(max_seconds))
    tw_len = int((tw_end - tw_start).total_seconds())
    # ensure the duration does not exceed the window length
    max_len = min(max_seconds, tw_len)
    min_len = min(min_seconds, max_len)
    duration = rng.randint(min_len, max_len)
    offset_max = tw_len - duration
    offset = rng.randint(0, offset_max) if offset_max > 0 else 0
    start_dt = tw_start + timedelta(seconds=offset)
    end_dt = start_dt + timedelta(seconds=duration)
    return start_dt, end_dt

def overlap_interval(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> Tuple[datetime, datetime] | None:
    """
    Compute the overlapping interval between two time ranges [a_start, a_end] and [b_start, b_end].
    Returns a (start, end) tuple if they overlap, otherwise returns None.
    """
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start < end:
        return start, end
    return None


def choice_weighted(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


# -----------------------------------------------------------------------------
# Utility functions for detection metadata
#
# To more closely mirror how video analytics systems record detections, we add
# simple helpers for confidence and bounding box generation. These functions
# return values that are written directly to CSV. Importantly, we do not
# convert the timestamp fields to strings here; we rely on Python's csv module
# to serialise datetime objects in ISO format (YYYY‑MM‑DD HH:MM:SS).

def random_confidence(rng: random.Random) -> float:
    """
    Generate a random confidence score between 0.7 and 0.99. Returning a float
    rather than string allows the CSV writer to serialise as needed.
    """
    return round(rng.uniform(0.7, 0.99), 3)


def random_bbox_json(rng: random.Random) -> str:
    """
    Generate a simple bounding box JSON string. Coordinates are arbitrary but
    within a 1920x1080 frame. The JSON is returned as a string because CSV
    cannot serialise dicts directly.
    """
    x = rng.randint(0, 1400)
    y = rng.randint(0, 800)
    w = rng.randint(50, 300)
    h = rng.randint(80, 400)
    return json.dumps({"x": x, "y": y, "w": w, "h": h})


def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


class CsvAppender:
    def __init__(self, path: str, header: List[str]):
        ensure_dir(os.path.dirname(path) or ".")
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self.count = 0

    def writerow(self, row: List[str]) -> None:
        self._w.writerow(row)
        self.count += 1

    def close(self) -> None:
        self._f.close()


def fmt_loc(i: int) -> str:
    return f"L{i:03d}"


def fmt_cam(i: int) -> str:
    return f"C{i:03d}"


def fmt_tw4(i: int) -> str:
    return f"{i:04d}"


def tw_id_str(tw_in_day: int) -> str:
    return f"TW{fmt_tw4(tw_in_day)}"


def mk_daily_entity_id(prefix: str, day_dt: datetime, seq: int) -> str:
    return f"{prefix}{yyyymmdd(day_dt)}_{seq:04d}"


def make_video_id(camera_id: str, day_dt: datetime, start_dt: datetime) -> str:
    return f"V_{camera_id}_{yyyymmdd(day_dt)}_{start_dt.strftime('%H%M%S')}"


def sample_excluding(rng: random.Random, pool: List[str], exclude: List[str], k: int) -> List[str]:
    if k <= 0:
        return []
    excluded = set(exclude)
    available = [x for x in pool if x not in excluded]
    if not available:
        return []
    return rng.sample(available, min(k, len(available)))


def partition_dir_name(partition_id: int) -> str:
    return f"partition_{partition_id:04d}"


@dataclass
class DensityCfg:
    persons_min: int
    persons_max: int
    things_per_person_mean: float
    vehicles_min: int
    vehicles_max: int
    stay_rate: float
    next_to_degree: int
    interact_ratio: float
    interact_cluster_min: int
    interact_cluster_max: int
    uses_ratio_per_vehicle: float
    uses_persons_min: int
    uses_persons_max: int


def density_profile(name: str) -> Dict[str, DensityCfg]:
    if name == "peak":
        return {
            "Outdoor": DensityCfg(60, 120, 0.70, 10, 40, 0.25, 6, 0.18, 2, 6, 0.55, 1, 3),
            "Entrance": DensityCfg(50, 110, 0.70, 6, 25, 0.30, 6, 0.16, 2, 6, 0.45, 1, 3),
            "Office": DensityCfg(20, 60, 0.55, 1, 8, 0.45, 5, 0.12, 2, 5, 0.25, 1, 2),
            "Corridor": DensityCfg(25, 70, 0.55, 1, 10, 0.40, 5, 0.12, 2, 5, 0.25, 1, 2),
            "Garage": DensityCfg(10, 35, 0.45, 10, 30, 0.55, 4, 0.08, 2, 5, 0.70, 1, 4),
            "Parking": DensityCfg(12, 40, 0.45, 10, 35, 0.55, 4, 0.08, 2, 5, 0.70, 1, 4),
        }
    if name == "heavy":
        return {
            "Outdoor": DensityCfg(35, 80, 0.65, 4, 18, 0.35, 5, 0.14, 2, 6, 0.45, 1, 3),
            "Entrance": DensityCfg(30, 75, 0.65, 3, 15, 0.38, 5, 0.14, 2, 6, 0.35, 1, 3),
            "Office": DensityCfg(12, 35, 0.50, 1, 6, 0.55, 4, 0.10, 2, 5, 0.20, 1, 2),
            "Corridor": DensityCfg(14, 40, 0.50, 1, 8, 0.50, 4, 0.10, 2, 5, 0.20, 1, 2),
            "Garage": DensityCfg(6, 18, 0.40, 6, 18, 0.65, 3, 0.06, 2, 5, 0.55, 1, 4),
            "Parking": DensityCfg(6, 20, 0.40, 6, 22, 0.65, 3, 0.06, 2, 5, 0.55, 1, 4),
        }
    return {
        "Outdoor": DensityCfg(20, 55, 0.60, 1, 10, 0.40, 4, 0.10, 2, 6, 0.30, 1, 2),
        "Entrance": DensityCfg(18, 50, 0.60, 1, 8, 0.45, 4, 0.10, 2, 6, 0.25, 1, 2),
        "Office": DensityCfg(8, 22, 0.45, 0, 4, 0.60, 3, 0.06, 2, 5, 0.15, 1, 2),
        "Corridor": DensityCfg(10, 25, 0.45, 0, 5, 0.55, 3, 0.06, 2, 5, 0.15, 1, 2),
        "Garage": DensityCfg(4, 12, 0.35, 3, 10, 0.70, 2, 0.04, 2, 4, 0.55, 1, 3),
        "Parking": DensityCfg(4, 14, 0.35, 3, 12, 0.70, 2, 0.04, 2, 4, 0.55, 1, 3),
    }


def gen_locations(rng: random.Random, n: int) -> List[Tuple[str, str, str]]:
    loc_types = [
        ("Outdoor", 0.25),
        ("Road", 0.20),
        ("Office", 0.20),
        ("Indoor", 0.15),
        ("Parking", 0.12),
        ("Garage", 0.08),
    ]
    out = []
    for i in range(1, n + 1):
        lt = choice_weighted(rng, loc_types)
        out.append((fmt_loc(i), f"{lt}_{i}", lt))
    return out


def gen_cameras(rng: random.Random, locations: List[Tuple[str, str, str]], cams_per_loc: int):
    view_types = ["Wide", "Corridor", "Entrance", "Zoom"]
    cams = []
    c = 1
    for loc_id, loc_name, loc_type in locations:
        is_indoor = "true" if loc_type in INDOOR_LOC_TYPES else "false"
        for j in range(cams_per_loc):
            cam_id = fmt_cam(c)
            view = rng.choice(view_types)
            cam_name = f"Cam_{loc_name}_{j+1}"
            cams.append((cam_id, cam_name, view, is_indoor, loc_id))
            c += 1
    return cams


def gen_daily_people(rng: random.Random, day_dt: datetime, n: int) -> List[List[str]]:
    genders = [("Male", 0.52), ("Female", 0.48)]
    age_groups = [("Child", 0.08), ("Adult", 0.78), ("Senior", 0.14)]
    rows = []
    for seq in range(1, n + 1):
        pid = mk_daily_entity_id("P", day_dt, seq)
        rows.append([pid, choice_weighted(rng, genders), choice_weighted(rng, age_groups)])
    return rows


def gen_daily_things(rng: random.Random, day_dt: datetime, n: int) -> List[List[str]]:
    thing_types = [("Backpack", 0.40), ("Handbag", 0.25), ("Bag", 0.20), ("Suitcase", 0.10), ("Box", 0.05)]
    sizes = [("Small", 0.35), ("Medium", 0.45), ("Large", 0.20)]
    colors = ["Black", "Blue", "Gray", "Red", "Green", "White", "Brown"]
    rows = []
    for seq in range(1, n + 1):
        tid = mk_daily_entity_id("T", day_dt, seq)
        rows.append([tid, choice_weighted(rng, thing_types), choice_weighted(rng, sizes), rng.choice(colors)])
    return rows


def gen_daily_vehicles(rng: random.Random, day_dt: datetime, n: int) -> List[List[str]]:
    vtypes = [("Car", 0.65), ("Motorbike", 0.25), ("Bus", 0.05), ("Truck", 0.05)]
    colors = ["White", "Black", "Gray", "Red", "Blue"]
    rows = []
    for seq in range(1, n + 1):
        vid = mk_daily_entity_id("VH", day_dt, seq)
        rows.append([vid, choice_weighted(rng, vtypes), rng.choice(colors)])
    return rows


@dataclass
class VideoRow:
    video_id: str
    camera_id: str
    partition_id: int
    date: str
    start_time: datetime
    end_time: datetime
    fps: int
    resolution: str


class PartitionWriters:
    def __init__(self, out_root: str, headers: Dict[str, List[str]]):
        self.out_root = out_root
        self.headers = headers
        self.by_partition_root = os.path.join(out_root, "by_partition")
        ensure_dir(self.by_partition_root)
        self._writers: Dict[int, Dict[str, CsvAppender]] = {}

    def _partition_path(self, partition_id: int) -> str:
        return os.path.join(self.by_partition_root, partition_dir_name(partition_id))

    def init_partition_static_files(
        self,
        partition_id: int,
        loc_row: List[str],
        camera_rows: List[List[str]],
        timewindow_rows: List[List[str]],
    ) -> None:
        pdir = self._partition_path(partition_id)
        ensure_dir(pdir)
        write_csv(os.path.join(pdir, "nodes_location.csv"), self.headers["nodes_location.csv"], [loc_row])
        write_csv(os.path.join(pdir, "nodes_camera.csv"), self.headers["nodes_camera.csv"], camera_rows)
        write_csv(os.path.join(pdir, "partitions.csv"), self.headers["partitions.csv"], [[str(partition_id), loc_row[0]]])
        write_csv(os.path.join(pdir, "nodes_timewindow.csv"), self.headers["nodes_timewindow.csv"], timewindow_rows)

    def get_partition_writer(self, partition_id: int, filename: str) -> CsvAppender:
        if partition_id not in self._writers:
            self._writers[partition_id] = {}
        if filename not in self._writers[partition_id]:
            pdir = self._partition_path(partition_id)
            ensure_dir(pdir)
            self._writers[partition_id][filename] = CsvAppender(os.path.join(pdir, filename), self.headers[filename])
        return self._writers[partition_id][filename]

    def writerow(self, partition_id: int, filename: str, row: List[str]) -> None:
        self.get_partition_writer(partition_id, filename).writerow(row)

    def close(self) -> None:
        for m in self._writers.values():
            for w in m.values():
                w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_locations", type=int, default=10)
    ap.add_argument("--cameras_per_location", type=int, default=3)
    ap.add_argument("--persons_pool", type=int, default=2000)
    ap.add_argument("--things_pool", type=int, default=800)
    ap.add_argument("--vehicles_pool", type=int, default=400)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--day_start", default="07:00:00")
    ap.add_argument("--day_seconds", type=int, default=24 * 3600)
    ap.add_argument("--tw_size_seconds", type=int, default=10)
    ap.add_argument("--video_duration_seconds", type=int, default=300)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--resolution", default="1920x1080")
    ap.add_argument("--density_profile", choices=["realistic", "heavy", "peak"], default="heavy")

    # Activity scaling: day/night multipliers and business hours
    ap.add_argument(
        "--day_active_hours",
        default="08:00-18:00",
        help="Time range for high activity (HH:MM-HH:MM). Outside this window, densities are scaled down.",
    )
    ap.add_argument(
        "--day_person_multiplier", type=float, default=1.5,
        help="Multiplier applied to person and vehicle counts during day active hours.",
    )
    ap.add_argument(
        "--night_person_multiplier", type=float, default=0.5,
        help="Multiplier applied to person and vehicle counts outside day active hours.",
    )
    args = ap.parse_args()

    if args.tw_size_seconds != TW_SECONDS_FIXED:
        print(f"[WARN] tw_size_seconds={args.tw_size_seconds} overridden to {TW_SECONDS_FIXED} (fixed TimeWindow).")
    args.tw_size_seconds = TW_SECONDS_FIXED

    if args.num_locations <= 0:
        raise ValueError("num_locations must be > 0")
    if args.cameras_per_location <= 0:
        raise ValueError("cameras_per_location must be > 0")
    if args.persons_pool <= 0:
        raise ValueError("persons_pool must be > 0")
    if args.things_pool < 0:
        raise ValueError("things_pool must be >= 0")
    if args.vehicles_pool < 0:
        raise ValueError("vehicles_pool must be >= 0")
    if args.days <= 0:
        raise ValueError("days must be > 0")
    if args.video_duration_seconds <= 0:
        raise ValueError("video_duration_seconds must be > 0")
    if args.day_seconds <= 0:
        raise ValueError("day_seconds must be > 0")
    if args.video_duration_seconds % args.tw_size_seconds != 0:
        raise ValueError("video_duration_seconds must be divisible by 10s (fixed TW)")
    if args.day_seconds % args.tw_size_seconds != 0:
        raise ValueError("day_seconds must be divisible by 10s (fixed TW)")
    if args.video_duration_seconds > args.day_seconds:
        raise ValueError("video_duration_seconds must be <= day_seconds")

    rng = random.Random(args.seed)

    # Parse day active hours into start and end times
    try:
        da_start_str, da_end_str = args.day_active_hours.split("-")
        da_start_h, da_start_m = map(int, da_start_str.split(":"))
        da_end_h, da_end_m = map(int, da_end_str.split(":"))
    except Exception:
        # Fallback to default 08:00-18:00 if parsing fails
        da_start_h, da_start_m, da_end_h, da_end_m = 8, 0, 18, 0
    ensure_dir(args.out)

    headers = {
        # Static nodes
        "nodes_location.csv": ["loc_id", "name", "loc_type"],
        "nodes_camera.csv": ["camera_id", "name", "view_type", "is_indoor"],
        "partitions.csv": ["partition_id", "loc_id"],
        "nodes_person.csv": ["pid", "gender", "age_group"],
        "nodes_thing.csv": ["tid", "thing_type", "size_category", "base_color"],
        "nodes_vehicle.csv": ["vid", "vehicle_type", "base_color"],
        "nodes_timewindow.csv": ["date", "tw_id", "start_time", "end_time", "duration_seconds"],
        "nodes_video.csv": ["video_id", "camera_id", "partition_id", "date", "start_time", "end_time", "fps", "resolution"],
        # Temporal nodes with dynamic attributes
        "nodes_person_TW.csv": [
            "pid_tw",      # composite id: person_id + TW
            "id_global",   # references nodes_person.csv
            "date",        # date part for partition pruning
            "tw_id",       # TimeWindow id (TWxxxx)
            "partition_id",# partition the node belongs to
            "shirt_color",
            "pant_color",
            "pose_state",
        ],
        "nodes_thing_TW.csv": [
            "tid_tw",
            "id_global",
            "date",
            "tw_id",
            "partition_id",
            "thing_type",
            "size_category",
            "base_color",
            "state",
        ],
        "nodes_vehicle_TW.csv": [
            "vid_tw",
            "id_global",
            "date",
            "tw_id",
            "partition_id",
            "vehicle_type",
            "base_color",
            "speed_kmh",
            "direction",
        ],
        # Relationship edges. Note: columns window_id and description are replaced
        # by more expressive date/tw_id/camera_id/location_id. The ts_start and
        # ts_end columns will be written as datetime objects (full timestamp).
        "rels.csv": [
            "source_id",
            "destination_id",
            "type",
            "ts_start",
            "ts_end",
            "date",
            "tw_id",
            "partition_id",
            "camera_id",
            "location_id",
            "confidence",
            "bbox",
            "description",
        ],
    }

    locations = gen_locations(rng, args.num_locations)
    for loc_id, _loc_name, loc_type in locations:
        if loc_type not in ALLOWED_LOC_TYPES:
            raise ValueError(f"Invalid loc_type={loc_type} for loc_id={loc_id}")
    loc_to_partition = {loc_id: i + 1 for i, (loc_id, _, _) in enumerate(locations)}
    loc_to_type = {loc_id: loc_type for (loc_id, _, loc_type) in locations}
    loc_rows_by_id = {loc_id: [loc_id, name, loc_type] for (loc_id, name, loc_type) in locations}


    cameras = gen_cameras(rng, locations, args.cameras_per_location)
    camera_to_indoor = {cam_id: (is_indoor == "true") for (cam_id, _cam_name, _view_type, is_indoor, _loc_id) in cameras}
    for cam_id, _cam_name, _view_type, is_indoor, loc_id in cameras:
        expected_indoor = loc_to_type[loc_id] in INDOOR_LOC_TYPES
        if (is_indoor == "true") != expected_indoor:
            raise ValueError(f"Camera indoor mismatch for camera_id={cam_id}, loc_id={loc_id}")

    # Build a mapping from location to cameras for easy cross‑camera movement.
    # We build this after cameras are generated to avoid referencing an undefined
    # variable earlier. The mapping is used later when computing cross‑camera
    # movement and when writing structural edges.
    from collections import defaultdict as _dd
    cams_by_loc: Dict[str, List[str]] = _dd(list)
    for cam_id, _cam_name, _view_type, _is_indoor, loc_id in cameras:
        cams_by_loc[loc_id].append(cam_id)

    # Build a simple undirected location graph: each location is adjacent to
    # its immediate predecessor and successor. This graph is used to allow
    # movement of persons/vehicles to neighbouring locations.
    location_graph: Dict[str, List[str]] = _dd(list)
    for i in range(1, args.num_locations):
        l1 = fmt_loc(i)
        l2 = fmt_loc(i + 1)
        location_graph[l1].append(l2)
        location_graph[l2].append(l1)
    cameras_by_partition: Dict[int, List[List[str]]] = defaultdict(list)
    for cam_id, cam_name, vtype, indoor, loc_id in cameras:
        cameras_by_partition[loc_to_partition[loc_id]].append([cam_id, cam_name, vtype, indoor])

    write_csv(os.path.join(args.out, "nodes_location.csv"), headers["nodes_location.csv"], [list(x) for x in locations])
    write_csv(os.path.join(args.out, "nodes_camera.csv"), headers["nodes_camera.csv"], [[cid, cname, vtype, indoor] for (cid, cname, vtype, indoor, _lid) in cameras])
    write_csv(os.path.join(args.out, "partitions.csv"), headers["partitions.csv"], [[str(loc_to_partition[loc_id]), loc_id] for (loc_id, _, _) in locations])

    day_start_h, day_start_m, day_start_s = map(int, args.day_start.split(":"))
    tw_per_day = args.day_seconds // args.tw_size_seconds
    video_per_day = args.day_seconds // args.video_duration_seconds
    tw_per_video = args.video_duration_seconds // args.tw_size_seconds

    dens_map = density_profile(args.density_profile)

    all_people_rows: List[List[str]] = []
    all_thing_rows: List[List[str]] = []
    all_vehicle_rows: List[List[str]] = []
    daily_people_pool: Dict[str, List[str]] = {}
    daily_thing_pool: Dict[str, List[str]] = {}
    daily_vehicle_pool: Dict[str, List[str]] = {}

    today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(args.days)]
    dates.reverse()  # past -> present, never future

    for day_date in dates:
        day_dt = datetime.combine(day_date, time(0, 0, 0))
        date_key = yyyy_mm_dd(day_dt)
        ppl = gen_daily_people(rng, day_dt, args.persons_pool)
        ths = gen_daily_things(rng, day_dt, args.things_pool)
        vhs = gen_daily_vehicles(rng, day_dt, args.vehicles_pool)
        all_people_rows.extend(ppl)
        all_thing_rows.extend(ths)
        all_vehicle_rows.extend(vhs)
        daily_people_pool[date_key] = [r[0] for r in ppl]
        daily_thing_pool[date_key] = [r[0] for r in ths]
        daily_vehicle_pool[date_key] = [r[0] for r in vhs]

    # Build lookup dictionaries for thing and vehicle attributes
    # These are used later when writing dynamic Thing_TW and Vehicle_TW rows.
    thing_attrs: Dict[str, Tuple[str, str, str]] = {}
    for tid, ttype, size_cat, base_color in all_thing_rows:
        thing_attrs[tid] = (ttype, size_cat, base_color)
    vehicle_attrs: Dict[str, Tuple[str, str]] = {}
    for vid, vtype, base_color in all_vehicle_rows:
        vehicle_attrs[vid] = (vtype, base_color)

    write_csv(os.path.join(args.out, "nodes_person.csv"), headers["nodes_person.csv"], all_people_rows)
    write_csv(os.path.join(args.out, "nodes_thing.csv"), headers["nodes_thing.csv"], all_thing_rows)
    write_csv(os.path.join(args.out, "nodes_vehicle.csv"), headers["nodes_vehicle.csv"], all_vehicle_rows)

    timewindow_rows: List[List[str]] = []
    for day_date in dates:
        day_base = datetime.combine(day_date, time(0, 0, 0))
        date_str = yyyy_mm_dd(day_base)
        t0 = datetime.combine(day_date, time(0, 0, 0))
        day_start_dt = day_base.replace(hour=day_start_h, minute=day_start_m, second=day_start_s)
        for tw_in_day in range(int(tw_per_day)):
            st = day_start_dt + timedelta(seconds=tw_in_day * TW_SECONDS_FIXED)
            en = st + timedelta(seconds=TW_SECONDS_FIXED)
            tw_idx = int((st - t0).total_seconds() // TW_SECONDS_FIXED)
            tw_id_val = f"TW{tw_idx:04d}"
            timewindow_rows.append([date_str, tw_id_val, hhmmss(st), hhmmss(en), str(TW_SECONDS_FIXED)])
    write_csv(os.path.join(args.out, "nodes_timewindow.csv"), headers["nodes_timewindow.csv"], timewindow_rows)

    partition_writers = PartitionWriters(args.out, headers)
    for loc_id, _name, _loc_type in locations:
        part = loc_to_partition[loc_id]
        partition_writers.init_partition_static_files(
            partition_id=part,
            loc_row=loc_rows_by_id[loc_id],
            camera_rows=cameras_by_partition.get(part, []),
            timewindow_rows=timewindow_rows,
        )

    videos: List[VideoRow] = []
    videos_by_cam_day: Dict[Tuple[str, str], List[VideoRow]] = defaultdict(list)
    for day_date in dates:
        day_base = datetime.combine(day_date, time(0, 0, 0))
        date_str = yyyy_mm_dd(day_base)
        day_start_dt = day_base.replace(hour=day_start_h, minute=day_start_m, second=day_start_s)
        for cam_id, _cam_name, _view_type, _is_indoor, loc_id in cameras:
            part = loc_to_partition[loc_id]
            for k in range(int(video_per_day)):
                st = day_start_dt + timedelta(seconds=k * args.video_duration_seconds)
                en = st + timedelta(seconds=args.video_duration_seconds)
                if not st < en:
                    raise ValueError(f"Invalid video interval for camera_id={cam_id}, date={date_str}")
                vid = make_video_id(cam_id, day_base, st)
                row = VideoRow(vid, cam_id, part, date_str, st, en, args.fps, args.resolution)
                videos.append(row)
                videos_by_cam_day[(date_str, cam_id)].append(row)

    videos.sort(key=lambda x: (x.date, x.start_time, x.camera_id))
    for k in videos_by_cam_day:
        videos_by_cam_day[k].sort(key=lambda x: x.start_time)

    with open(os.path.join(args.out, "nodes_video.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers["nodes_video.csv"])
        for v in videos:
            row = [v.video_id, v.camera_id, str(v.partition_id), v.date, hhmmss(v.start_time), hhmmss(v.end_time), str(v.fps), v.resolution]
            w.writerow(row)
            partition_writers.writerow(v.partition_id, "nodes_video.csv", row)

    person_tw_w = CsvAppender(os.path.join(args.out, "nodes_person_TW.csv"), headers["nodes_person_TW.csv"])
    thing_tw_w = CsvAppender(os.path.join(args.out, "nodes_thing_TW.csv"), headers["nodes_thing_TW.csv"])
    vehicle_tw_w = CsvAppender(os.path.join(args.out, "nodes_vehicle_TW.csv"), headers["nodes_vehicle_TW.csv"])
    rels_w = CsvAppender(os.path.join(args.out, "rels.csv"), headers["rels.csv"])

    if args.num_locations > 1:
        for i in range(1, args.num_locations):
            loc1 = fmt_loc(i)
            loc2 = fmt_loc(i + 1)
            p1 = loc_to_partition[loc1]
            p2 = loc_to_partition[loc2]
            # NEAR_BY is structural: no timestamps, camera/location context
            row = [
                loc1,
                loc2,
                "NEAR_BY",
                "",  # ts_start
                "",  # ts_end
                "",  # date
                "",  # tw_id
                str(p1),  # partition_id (loc1 determines partition)
                "",  # camera_id
                "",  # location_id
                "",  # confidence
                "",  # bbox
                "distance=50",  # description
            ]
            rels_w.writerow(row)
            partition_writers.writerow(p1, "rels.csv", row)
            reverse_row = [
                loc2,
                loc1,
                "NEAR_BY",
                "",
                "",
                "",
                "",
                str(p2),
                "",
                "",
                "",
                "",
                "distance=50",
            ]
            rels_w.writerow(reverse_row)
            partition_writers.writerow(p2, "rels.csv", reverse_row)

    for day_date in dates:
        date_str = yyyy_mm_dd(datetime.combine(day_date, time(0, 0, 0)))
        for cam_id, *_rest in cameras:
            vlist = videos_by_cam_day[(date_str, cam_id)]
            for i in range(1, len(vlist)):
                v1 = vlist[i - 1]
                v2 = vlist[i]
                boundary_time = v1.end_time
                boundary_end = boundary_time + timedelta(seconds=1)
                row = [
                    v1.video_id,
                    v2.video_id,
                    "NEXT_TO",
                    boundary_time,
                    boundary_end,
                    v1.date,
                    "",
                    str(v1.partition_id),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                rels_w.writerow(row)
                partition_writers.writerow(v1.partition_id, "rels.csv", row)

    shirt_colors = ["Blue", "Black", "Gray", "Red", "Green", "White", "Brown", "Yellow"]
    pant_colors = ["Black", "Blue", "Gray", "Brown", "White"]
    active_persons: Dict[Tuple[str, str], List[str]] = {}
    active_vehicles: Dict[Tuple[str, str], List[str]] = {}
    wrote_located_at = set()
    wrote_recorded_by = set()

    min_duration_seconds = 1
    for day_date in dates:
        day_base = datetime.combine(day_date, time(0, 0, 0))
        date_str = yyyy_mm_dd(day_base)
        t0 = datetime.combine(day_date, time(0, 0, 0))
        day_start_dt = day_base.replace(hour=day_start_h, minute=day_start_m, second=day_start_s)
        pool_p = daily_people_pool[date_str]
        pool_t = daily_thing_pool[date_str]
        pool_v = daily_vehicle_pool[date_str]
        # Maintain per-day persistent attributes for people and vehicles
        # person_clothes maps person id to (shirt_color, pant_color)
        person_clothes = {}
        # vehicle_prev maps vehicle id to (speed, direction)
        vehicle_prev = {}

        for tw_in_day in range(int(tw_per_day)):
            tw_start = day_start_dt + timedelta(seconds=tw_in_day * TW_SECONDS_FIXED)
            tw_end = tw_start + timedelta(seconds=TW_SECONDS_FIXED)
            video_idx = tw_in_day // tw_per_video
            if video_idx >= video_per_day:
                continue

            tw_idx = int((tw_start - t0).total_seconds() // TW_SECONDS_FIXED)
            tw_tag = f"{tw_idx:04d}"
            tw_key = f"TW{tw_idx:04d}"

            # Track which persons and vehicles have already been assigned to a camera in this TW
            assigned_persons = set()
            assigned_vehicles = set()

            for cam_id, _cam_name, _view_type, _is_indoor, loc_id in cameras:
                part = loc_to_partition[loc_id]
                loc_type = loc_to_type.get(loc_id, "Outdoor")
                cfg = dens_map.get(loc_type, dens_map["Outdoor"])
                vlist = videos_by_cam_day[(date_str, cam_id)]
                if video_idx >= len(vlist):
                    continue
                video = vlist[video_idx]

                if cam_id not in wrote_located_at:
                    # LOCATED_AT is structural: no timestamps; include camera_id and location_id in context
                    row_loc = [
                        cam_id,        # source (Camera)
                        loc_id,        # destination (Location)
                        "LOCATED_AT",
                        "",          # ts_start
                        "",          # ts_end
                        "",          # date
                        "",          # tw_id
                        str(part),     # partition_id
                        cam_id,        # camera_id
                        loc_id,        # location_id
                        "",          # confidence
                        "",          # bbox
                        "",          # description
                    ]
                    rels_w.writerow(row_loc)
                    partition_writers.writerow(part, "rels.csv", row_loc)
                    wrote_located_at.add(cam_id)

                if video.video_id not in wrote_recorded_by:
                    # RECORDED_BY: video segment recorded by a camera over its duration
                    row_rec = [
                        video.video_id,
                        cam_id,
                        "RECORDED_BY",
                        video.start_time,
                        video.end_time,
                        video.date,
                        "",
                        str(part),
                        cam_id,
                        loc_id,
                        "",
                        "",
                        "",
                    ]
                    rels_w.writerow(row_rec)
                    partition_writers.writerow(part, "rels.csv", row_rec)
                    wrote_recorded_by.add(video.video_id)

                # Determine whether the current time window is within high-activity hours
                curr_hour = tw_start.hour
                curr_min = tw_start.minute
                # Compare using (hour, minute)
                in_day_time = False
                # We parse day_active_hours at runtime; da_start_h, da_start_m, da_end_h, da_end_m
                if ((curr_hour > da_start_h) or (curr_hour == da_start_h and curr_min >= da_start_m)) and (
                    (curr_hour < da_end_h) or (curr_hour == da_end_h and curr_min < da_end_m)
                ):
                    in_day_time = True
                # Choose multiplier based on day/night
                if in_day_time:
                    mult = args.day_person_multiplier
                else:
                    mult = args.night_person_multiplier

                # Determine target numbers of persons and vehicles for this camera/time window
                base_p = rng.randint(cfg.persons_min, cfg.persons_max)
                base_v = rng.randint(cfg.vehicles_min, cfg.vehicles_max)
                # Compute number of persons to appear in this camera/time window based on
                # density profile and day/night multiplier. Cap at the size of the
                # available person pool for the day. If the density configuration
                # specifies a non‑zero interact_ratio, ensure that at least two people
                # are present (when possible) so that INTERACTS_WITH edges can be
                # generated. Without this guard, small pools or low multipliers
                # could result in zero or one person in a TW, which would make
                # the interaction ratio effectively zero across the dataset.
                target_persons = min(int(base_p * mult), len(pool_p))
                # If interactions are enabled and there are at least two people
                # available in the pool, guarantee at least two persons in this TW.
                if cfg.interact_ratio > 0 and target_persons < 2 and len(pool_p) >= 2:
                    target_persons = 2
                target_vehicles = min(int(base_v * mult), len(pool_v))
                if camera_to_indoor.get(cam_id, False) or loc_type in INDOOR_LOC_TYPES:
                    target_vehicles = 0

                # --- Cross‑camera movement logic ---
                # We maintain a dictionary of active persons per camera across time windows.
                # People may stay at the same camera (stay_rate), or move from neighbouring
                # cameras based on a small probability.  However, to ensure each person
                # appears in at most one camera per TimeWindow, we will remove any
                # candidate who has already been assigned to another camera in this TW
                key = (date_str, cam_id)
                prev_p = active_persons.get(key, [])
                # persons who stay at the same camera
                keep_n = clamp_int(int(len(prev_p) * cfg.stay_rate), 0, min(len(prev_p), target_persons))
                keep = rng.sample(prev_p, keep_n) if keep_n > 0 else []
                persons_present: List[str] = []
                persons_present.extend(keep)

                # Move persons from cameras in the same location (excluding current camera)
                move_fraction = 0.15  # 15% of people in neighbouring cameras move here
                for other_cam in cams_by_loc[loc_id]:
                    if other_cam == cam_id:
                        continue
                    prev_o = active_persons.get((day, other_cam), [])
                    move_n = int(len(prev_o) * move_fraction)
                    if move_n > 0:
                        movers = rng.sample(prev_o, min(move_n, len(prev_o)))
                        persons_present.extend(movers)
                        # remove moved persons from the other camera's active list so they don't duplicate
                        for m in movers:
                            try:
                                prev_o.remove(m)
                            except ValueError:
                                pass
                        active_persons[(day, other_cam)] = prev_o

                # Move persons from neighbouring locations' cameras
                for nbr_loc in location_graph.get(loc_id, []):
                    for nbr_cam in cams_by_loc.get(nbr_loc, []):
                        prev_n = active_persons.get((day, nbr_cam), [])
                        move_n2 = int(len(prev_n) * move_fraction)
                        if move_n2 > 0:
                            movers2 = rng.sample(prev_n, min(move_n2, len(prev_n)))
                            persons_present.extend(movers2)
                            for m in movers2:
                                try:
                                    prev_n.remove(m)
                                except ValueError:
                                    pass
                            active_persons[(day, nbr_cam)] = prev_n

                # Remove duplicates and persons already assigned to other cameras in this TW
                # Also maintain uniqueness within persons_present
                unique_present = []
                seen_in_camera = set()
                for pid in persons_present:
                    if pid not in seen_in_camera and pid not in assigned_persons:
                        unique_present.append(pid)
                        seen_in_camera.add(pid)
                persons_present = unique_present

                # Truncate if we already exceeded target
                if len(persons_present) > target_persons:
                    persons_present = persons_present[:target_persons]

                # Fill newcomers from global pool, avoiding duplicates and previously assigned persons
                newcomers = sample_excluding(rng, pool_p, list(seen_in_camera | assigned_persons), target_persons - len(persons_present))
                persons_present.extend(newcomers)
                # Update assigned persons so that subsequent cameras do not reuse them
                for pid in persons_present:
                    assigned_persons.add(pid)
                # Save current camera's active persons
                active_persons[key] = persons_present[:]

                # --- Vehicles with similar cross‑camera movement ---
                prev_vh = active_vehicles.get(key, [])
                keep_vh_n = clamp_int(int(len(prev_vh) * cfg.stay_rate), 0, min(len(prev_vh), target_vehicles))
                keep_vh = rng.sample(prev_vh, keep_vh_n) if keep_vh_n > 0 else []
                vehicles_present: List[str] = []
                vehicles_present.extend(keep_vh)
                # move vehicles from other cameras in same location
                for other_cam in cams_by_loc[loc_id]:
                    if other_cam == cam_id:
                        continue
                    prev_o_v = active_vehicles.get((day, other_cam), [])
                    move_v_n = int(len(prev_o_v) * move_fraction)
                    if move_v_n > 0:
                        movers_v = rng.sample(prev_o_v, min(move_v_n, len(prev_o_v)))
                        vehicles_present.extend(movers_v)
                        for m in movers_v:
                            try:
                                prev_o_v.remove(m)
                            except ValueError:
                                pass
                        active_vehicles[(day, other_cam)] = prev_o_v
                # move vehicles from neighbouring locations' cameras
                for nbr_loc in location_graph.get(loc_id, []):
                    for nbr_cam in cams_by_loc.get(nbr_loc, []):
                        prev_n_v = active_vehicles.get((day, nbr_cam), [])
                        move_v_n2 = int(len(prev_n_v) * move_fraction)
                        if move_v_n2 > 0:
                            movers_v2 = rng.sample(prev_n_v, min(move_v_n2, len(prev_n_v)))
                            vehicles_present.extend(movers_v2)
                            for m in movers_v2:
                                try:
                                    prev_n_v.remove(m)
                                except ValueError:
                                    pass
                            active_vehicles[(day, nbr_cam)] = prev_n_v
                # Remove duplicates and vehicles already assigned to other cameras in this TW
                unique_v_present = []
                seen_v_in_cam = set()
                for vid in vehicles_present:
                    if vid not in seen_v_in_cam and vid not in assigned_vehicles:
                        unique_v_present.append(vid)
                        seen_v_in_cam.add(vid)
                vehicles_present = unique_v_present
                # Truncate if we already exceeded target
                if len(vehicles_present) > target_vehicles:
                    vehicles_present = vehicles_present[:target_vehicles]
                # Fill newcomers from global pool, avoiding duplicates and previously assigned vehicles
                new_vh = sample_excluding(rng, pool_v, list(seen_v_in_cam | assigned_vehicles), target_vehicles - len(vehicles_present))
                vehicles_present.extend(new_vh)
                # Update assigned vehicles so that subsequent cameras do not reuse them
                for vid in vehicles_present:
                    assigned_vehicles.add(vid)
                active_vehicles[key] = vehicles_present[:]

                # Number of things is proportional to number of persons present
                exp_things = int(len(persons_present) * cfg.things_per_person_mean)
                exp_things = clamp_int(exp_things + rng.randint(-2, 3), 0, exp_things + 10)
                exp_things = min(exp_things, len(pool_t))
                things_present = rng.sample(pool_t, exp_things) if exp_things > 0 else []

                # Initialize dictionaries to store detection intervals for this time window
                person_intervals: Dict[str, Tuple[datetime, datetime]] = {}
                thing_intervals: Dict[str, Tuple[datetime, datetime]] = {}
                vehicle_intervals: Dict[str, Tuple[datetime, datetime]] = {}

                # To encourage non zero interaction ratios, assign overlapping detection
                # intervals to at least two persons when interactions are enabled and there
                # are at least two persons present. This ensures a temporal overlap exists
                # for a subset of persons, satisfying the edge generation rules.
                overlap_start = None  # type: Optional[datetime]
                overlap_end = None    # type: Optional[datetime]
                if cfg.interact_ratio > 0 and len(persons_present) >= 2:
                    tw_len_sec = int((tw_end - tw_start).total_seconds())
                    base_dur = rng.randint(2, min(8, tw_len_sec))
                    offset_max = tw_len_sec - base_dur
                    base_offset = rng.randint(0, offset_max) if offset_max > 0 else 0
                    overlap_start = tw_start + timedelta(seconds=base_offset)
                    overlap_end = overlap_start + timedelta(seconds=base_dur)
                    rng.shuffle(persons_present)

                person_tw_ids: List[str] = []
                for idx, pid in enumerate(persons_present):
                    # Build Person-TW ID
                    pid_tw = f"{pid}_TW{tw_tag}"
                    person_tw_ids.append(pid_tw)
                    # Assign detection interval; first two persons get the shared overlap
                    if overlap_start is not None and idx < 2:
                        p_det_start, p_det_end = overlap_start, overlap_end
                    else:
                        p_det_start, p_det_end = sample_interval_within_tw(rng, tw_start, tw_end, 2, 8)
                    if (p_det_end - p_det_start).total_seconds() < min_duration_seconds:
                        p_det_end = min(tw_end, p_det_start + timedelta(seconds=min_duration_seconds))
                    person_intervals[pid_tw] = (p_det_start, p_det_end)
                    # Write Person-TW node with dynamic attributes (shirt, pant, pose)
                    pose_states = ["Standing", "Walking", "Running", "Sitting"]
                    # Determine shirt and pant colors with memory
                    # Probability that a person changes clothes during the day (e.g., to disguise)
                    cloth_change_prob = 0.03
                    if (pid not in person_clothes) or (rng.random() < cloth_change_prob):
                        shirt = rng.choice(shirt_colors)
                        pant = rng.choice(pant_colors)
                        person_clothes[pid] = (shirt, pant)
                    else:
                        shirt, pant = person_clothes[pid]
                    row = [
                        pid_tw,
                        pid,
                        date_str,
                        tw_key,
                        str(part),
                        shirt,
                        pant,
                        rng.choice(pose_states),
                    ]
                    person_tw_w.writerow(row)
                    partition_writers.writerow(part, "nodes_person_TW.csv", row)
                    # Write DETECTED_IN relation with full datetime and metadata
                    det_row = [
                        pid_tw,                 # source_id
                        video.video_id,         # destination_id
                        "DETECTED_IN",          # type
                        p_det_start,            # ts_start (datetime)
                        p_det_end,              # ts_end (datetime)
                        date_str,               # date
                        tw_key,                 # tw_id
                        str(part),              # partition_id
                        cam_id,                 # camera_id
                        loc_id,                 # location_id
                        random_confidence(rng), # confidence
                        random_bbox_json(rng),  # bbox
                        "",                    # description
                    ]
                    rels_w.writerow(det_row)
                    partition_writers.writerow(part, "rels.csv", det_row)

                thing_tw_ids: List[str] = []
                for tid in things_present:
                    tid_tw = f"{tid}_TW{tw_tag}"
                    thing_tw_ids.append(tid_tw)
                    # Random detection interval for thing
                    t_det_start, t_det_end = sample_interval_within_tw(rng, tw_start, tw_end, 2, 8)
                    if (t_det_end - t_det_start).total_seconds() < min_duration_seconds:
                        t_det_end = min(tw_end, t_det_start + timedelta(seconds=min_duration_seconds))
                    thing_intervals[tid_tw] = (t_det_start, t_det_end)
                    # Retrieve static attributes for thing
                    ttype, size_cat, base_color = thing_attrs.get(tid, ("Bag", "Medium", "Black"))
                    state = rng.choice(["Carried", "Stationary"])
                    row = [
                        tid_tw,    # id
                        tid,       # global id
                        date_str,
                        tw_key,
                        str(part),
                        ttype,
                        size_cat,
                        base_color,
                        state,
                    ]
                    thing_tw_w.writerow(row)
                    partition_writers.writerow(part, "nodes_thing_TW.csv", row)
                    # DETECTED_IN edge
                    det_row = [
                        tid_tw,
                        video.video_id,
                        "DETECTED_IN",
                        t_det_start,
                        t_det_end,
                        date_str,
                        tw_key,
                        str(part),
                        cam_id,
                        loc_id,
                        random_confidence(rng),
                        random_bbox_json(rng),
                        "",
                    ]
                    rels_w.writerow(det_row)
                    partition_writers.writerow(part, "rels.csv", det_row)

                vehicle_tw_ids: List[str] = []
                for vh in vehicles_present:
                    vh_tw = f"{vh}_TW{tw_tag}"
                    vehicle_tw_ids.append(vh_tw)
                    # Random detection interval for vehicle
                    v_det_start, v_det_end = sample_interval_within_tw(rng, tw_start, tw_end, 2, 8)
                    if (v_det_end - v_det_start).total_seconds() < min_duration_seconds:
                        v_det_end = min(tw_end, v_det_start + timedelta(seconds=min_duration_seconds))
                    vehicle_intervals[vh_tw] = (v_det_start, v_det_end)
                    # Lookup vehicle static attributes
                    vtype, vcolor = vehicle_attrs.get(vh, ("Car", "White"))
                    # Determine speed and direction with memory
                    # Probability that vehicle makes a large change in speed/direction
                    vehicle_change_prob = 0.05
                    big_change = False
                    # If this vehicle has no previous record or we decide on a big change, sample new
                    if vh not in vehicle_prev or rng.random() < vehicle_change_prob:
                        big_change = True
                        speed = rng.randint(0, 60)
                        direction = rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
                    else:
                        prev_speed, prev_dir = vehicle_prev[vh]
                        # With high probability, vary speed slightly around previous value
                        # Variation ±5 km/h, clamped to 0–60
                        delta = rng.randint(-5, 5)
                        speed = max(0, min(60, prev_speed + delta))
                        # Direction list for circular indexing
                        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                        idx = dirs.index(prev_dir)
                        # Small change: stay same or move to adjacent direction
                        dir_delta = rng.choice([-1, 0, 1])
                        direction = dirs[(idx + dir_delta) % len(dirs)]
                    # Save current speed and direction for next TW
                    vehicle_prev[vh] = (speed, direction)
                    row = [
                        vh_tw,
                        vh,
                        date_str,
                        tw_key,
                        str(part),
                        vtype,
                        vcolor,
                        str(speed),
                        direction,
                    ]
                    vehicle_tw_w.writerow(row)
                    partition_writers.writerow(part, "nodes_vehicle_TW.csv", row)
                    det_row = [
                        vh_tw,
                        video.video_id,
                        "DETECTED_IN",
                        v_det_start,
                        v_det_end,
                        date_str,
                        tw_key,
                        str(part),
                        cam_id,
                        loc_id,
                        random_confidence(rng),
                        random_bbox_json(rng),
                        "",
                    ]
                    rels_w.writerow(det_row)
                    partition_writers.writerow(part, "rels.csv", det_row)

                if person_tw_ids and thing_tw_ids:
                    # Assign carriers for things and compute overlapping intervals
                    for tid_tw in thing_tw_ids:
                        carrier = rng.choice(person_tw_ids)
                        p_start, p_end = person_intervals.get(carrier, (tw_start, tw_end))
                        t_start, t_end = thing_intervals.get(tid_tw, (tw_start, tw_end))
                        overlap = overlap_interval(p_start, p_end, t_start, t_end)
                        # Only generate the relation if there is a temporal overlap
                        if not overlap:
                            continue
                        rel_start, rel_end = overlap
                        if (rel_end - rel_start).total_seconds() < min_duration_seconds:
                            continue
                        row = [
                            carrier,                 # source (Person_TW)
                            tid_tw,                  # dest (Thing_TW)
                            "CARRIES",             # type
                            rel_start,               # ts_start (datetime)
                            rel_end,                 # ts_end (datetime)
                            date_str,                # date
                            tw_key,                  # tw_id
                            str(part),               # partition_id
                            "",                    # camera_id (semantic edge)
                            "",                    # location_id (semantic edge)
                            random_confidence(rng),  # confidence
                            "",                    # bbox (optional)
                            "",                    # description
                        ]
                        rels_w.writerow(row)
                        partition_writers.writerow(part, "rels.csv", row)

                if person_tw_ids and vehicle_tw_ids:
                    for vh_tw in vehicle_tw_ids:
                        if rng.random() <= cfg.uses_ratio_per_vehicle:
                            k = clamp_int(rng.randint(cfg.uses_persons_min, cfg.uses_persons_max), 1, min(len(person_tw_ids), 10))
                            for u in rng.sample(person_tw_ids, k):
                                p_start, p_end = person_intervals.get(u, (tw_start, tw_end))
                                v_start, v_end = vehicle_intervals.get(vh_tw, (tw_start, tw_end))
                                overlap = overlap_interval(p_start, p_end, v_start, v_end)
                                # Only generate the relation if there is a temporal overlap
                                if not overlap:
                                    continue
                                rel_start, rel_end = overlap
                                if (rel_end - rel_start).total_seconds() < min_duration_seconds:
                                    continue
                                row = [
                                    u,
                                    vh_tw,
                                    "USES",
                                    rel_start,
                                    rel_end,
                                    date_str,
                                    tw_key,
                                    str(part),
                                    "",                    # camera_id
                                    "",                    # location_id
                                    random_confidence(rng),
                                    "",                    # bbox
                                    "",                    # description
                                ]
                                rels_w.writerow(row)
                                partition_writers.writerow(part, "rels.csv", row)
                # Add INTERACTS_WITH relationships between persons in this TW
                if len(person_tw_ids) >= 2 and cfg.interact_ratio > 0:
                    # Use interact ratio directly (not scaled by person multiplier)
                    curr_ir = cfg.interact_ratio
                    # Cap ratio to 1.0
                    if curr_ir > 1.0:
                        curr_ir = 1.0
                    n = len(person_tw_ids)
                    # Total possible unique pairs among persons in this time window
                    total_pairs = n * (n - 1) // 2
                    # Compute desired number of interacting pairs based on ratio
                    desired_pairs = int(total_pairs * curr_ir)
                    # Ensure at least interact_cluster_min pairs
                    if desired_pairs < cfg.interact_cluster_min:
                        desired_pairs = cfg.interact_cluster_min
                    # Clamp to total possible pairs (no upper cluster limit)
                    if desired_pairs > total_pairs:
                        desired_pairs = total_pairs
                    # Only proceed if we need at least one interaction
                    if desired_pairs > 0:
                        seen_pairs = set()
                        attempts = 0
                        max_attempts = total_pairs * 5 if total_pairs > 0 else 0
                        while len(seen_pairs) < desired_pairs and attempts < max_attempts:
                            attempts += 1
                            a, b = rng.sample(person_tw_ids, 2)
                            pair = tuple(sorted((a, b)))
                            if pair in seen_pairs:
                                continue
                            a_start, a_end = person_intervals.get(pair[0], (tw_start, tw_end))
                            b_start, b_end = person_intervals.get(pair[1], (tw_start, tw_end))
                            overlap = overlap_interval(a_start, a_end, b_start, b_end)
                            # Only create relation if there is a temporal overlap
                            if not overlap:
                                continue
                            rel_start, rel_end = overlap
                            if (rel_end - rel_start).total_seconds() < min_duration_seconds:
                                continue
                            row = [
                                pair[0],
                                pair[1],
                                "INTERACTS_WITH",
                                rel_start,
                                rel_end,
                                date_str,
                                tw_key,
                                str(part),
                                "",                    # camera_id
                                "",                    # location_id
                                random_confidence(rng),
                                "",                    # bbox
                                "",                    # description
                            ]
                            rels_w.writerow(row)
                            partition_writers.writerow(part, "rels.csv", row)
                            seen_pairs.add(pair)

    # Close all CSV writers to flush buffered rows
    person_tw_w.close()
    thing_tw_w.close()
    vehicle_tw_w.close()
    rels_w.close()
    # Close partition writers to flush partition files
    partition_writers.close()

    # ----------------------------------------------------------------------
    # Post‑processing step to ensure that root dynamic node files contain all
    # rows that were written into partition files.  In some rare cases the
    # root CSV writers may miss a small number of rows due to buffering or
    # execution order, which leads to a mismatch between the row counts
    # reported in the root directory and the sum of rows in the per‑partition
    # subdirectories.  To guarantee consistency, we load the per‑partition
    # dynamic node files (Person_TW, Thing_TW, Vehicle_TW) and append any
    # rows not already present in the corresponding root file.  This
    # operation is idempotent and does not introduce duplicates because
    # we maintain a set of existing identifiers.
    def append_missing_rows(root_path: str, part_subdir: str, id_col: int) -> None:
        """
        Ensure that every row present in any partition file for a given
        dynamic CSV also exists in the root file.  Arguments:
            root_path: absolute path to the root CSV file
            part_subdir: filename within each partition directory
            id_col: index of the identifier column in the CSV rows
        The function reads the existing root rows into a set of IDs, then
        iterates through every partition file of the same name.  If a row
        has an ID not present in the root set, the row is appended to
        the root CSV.  This preserves all partition data in the root.
        """
        # Load existing rows into memory and capture header
        if not os.path.exists(root_path):
            return
        with open(root_path, newline="", encoding="utf-8") as rf:
            reader = list(csv.reader(rf))
            if not reader:
                return
            header = reader[0]
            existing_rows = reader[1:]
        existing_ids = {row[id_col] for row in existing_rows}
        new_rows = []
        part_root = os.path.join(args.out, "by_partition")
        if os.path.isdir(part_root):
            for part in os.listdir(part_root):
                part_path = os.path.join(part_root, part, part_subdir)
                if not os.path.exists(part_path):
                    continue
                with open(part_path, newline="", encoding="utf-8") as pf:
                    preader = csv.reader(pf)
                    # Skip header
                    try:
                        next(preader)
                    except StopIteration:
                        continue
                    for prow in preader:
                        if len(prow) <= id_col:
                            continue
                        pid = prow[id_col]
                        # Append to new_rows if not already in root
                        if pid not in existing_ids:
                            new_rows.append(prow)
                            existing_ids.add(pid)
        # If there are new rows, append them to the root file
        if new_rows:
            with open(root_path, "a", newline="", encoding="utf-8") as rf:
                writer = csv.writer(rf)
                for r in new_rows:
                    writer.writerow(r)

    # Apply the consolidation for dynamic node files
    append_missing_rows(
        os.path.join(args.out, "nodes_person_TW.csv"), "nodes_person_TW.csv", id_col=0
    )
    append_missing_rows(
        os.path.join(args.out, "nodes_thing_TW.csv"), "nodes_thing_TW.csv", id_col=0
    )
    append_missing_rows(
        os.path.join(args.out, "nodes_vehicle_TW.csv"), "nodes_vehicle_TW.csv", id_col=0
    )

    # Reprint summary information after consolidation
    print("DONE")
    print(f"TimeWindow fixed = {TW_SECONDS_FIXED}s. Root CSVs kept unchanged at: {args.out}")
    print(f"Partition folders created at: {os.path.join(args.out, 'by_partition')}")
    print(f"days={args.days}, day_seconds={args.day_seconds}, tw_per_day={tw_per_day}")
    print(f"video_duration={args.video_duration_seconds}s, video_per_day={video_per_day}, tw_per_video={tw_per_video}")


if __name__ == "__main__":
    main()
