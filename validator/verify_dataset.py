#!/usr/bin/env python3
"""
Publication-level validator for VideoGraphDB CSV datasets.

This validator is designed for the current generator layout:
- root flat CSV files (nodes_*.csv, rels.csv, partitions.csv)
- optional by_partition/partition_xxxx/*.csv layout

It validates:
1. Header/schema compliance for known CSVs
2. Uniqueness of all node identifiers
3. Referential integrity of edges
4. TimeWindow bucket correctness and interval sanity
5. Video coverage for DETECTED_IN and semantic edges
6. Exactly one DETECTED_IN per Entity_TW
7. Overlap validity for CARRIES / USES / INTERACTS_WITH
8. Undirected consistency for NEAR_BY and INTERACTS_WITH
9. Semantic constraints (e.g. Vehicle not in Indoor / Office)
10. Partition integrity and row-count preservation
11. Duplicate edge detection
12. Query-readiness signals (interaction ratio, movement, cross-location behavior)

Exit code:
- 0 if validation passes
- 1 if validation fails
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


TEMPORAL_EDGE_TYPES = {"DETECTED_IN", "CARRIES", "USES", "INTERACTS_WITH"}
STRUCTURAL_EDGE_TYPES = {"LOCATED_AT", "RECORDED_BY", "NEAR_BY", "NEXT_TO"}
ALL_EDGE_TYPES = TEMPORAL_EDGE_TYPES | STRUCTURAL_EDGE_TYPES
DYNAMIC_NODE_FILES = ["nodes_person_TW.csv", "nodes_thing_TW.csv", "nodes_vehicle_TW.csv"]
ROOT_FILES_TO_PARTITION_CHECK = [
    "nodes_location.csv",
    "nodes_camera.csv",
    "partitions.csv",
    "nodes_timewindow.csv",
    "nodes_video.csv",
    "nodes_person_TW.csv",
    "nodes_thing_TW.csv",
    "nodes_vehicle_TW.csv",
    "rels.csv",
]


EXPECTED_HEADERS = {
    "nodes_location.csv": ["loc_id", "name", "loc_type"],
    "nodes_camera.csv": ["camera_id", "name", "view_type", "is_indoor"],
    "partitions.csv": ["partition_id", "loc_id"],
    "nodes_person.csv": ["pid", "gender", "age_group"],
    "nodes_thing.csv": ["tid", "thing_type", "size_category", "base_color"],
    "nodes_vehicle.csv": ["vid", "vehicle_type", "base_color"],
    "nodes_timewindow.csv": ["date", "tw_id", "start_time", "end_time", "duration_seconds"],
    "nodes_video.csv": ["video_id", "camera_id", "partition_id", "date", "start_time", "end_time", "fps", "resolution"],
    "nodes_person_TW.csv": ["pid_tw", "id_global", "date", "tw_id", "partition_id", "shirt_color", "pant_color", "pose_state"],
    "nodes_thing_TW.csv": ["tid_tw", "id_global", "date", "tw_id", "partition_id", "thing_type", "size_category", "base_color", "state"],
    "nodes_vehicle_TW.csv": ["vid_tw", "id_global", "date", "tw_id", "partition_id", "vehicle_type", "base_color", "speed_kmh", "direction"],
    "rels.csv": ["source_id", "destination_id", "type", "ts_start", "ts_end", "date", "tw_id", "partition_id", "camera_id", "location_id", "confidence", "bbox", "description"],
}


class ValidationError(RuntimeError):
    pass


@dataclass
class IssueCollector:
    fail_fast: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        if self.fail_fast:
            raise ValidationError(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ValidationSummary:
    entity_valid: str = "PASS"
    detection_valid: str = "PASS"
    relation_valid: str = "PASS"
    partition_valid: str = "PASS"
    schema_valid: str = "PASS"
    semantic_valid: str = "PASS"
    uniqueness_valid: str = "PASS"
    referential_valid: str = "PASS"
    query_readiness_valid: str = "PASS"
    interaction_ratio_per_tw: float = 0.0
    multi_location_entities: int = 0
    cross_partition_entities: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_valid": self.entity_valid,
            "detection_valid": self.detection_valid,
            "relation_valid": self.relation_valid,
            "partition_valid": self.partition_valid,
            "schema_valid": self.schema_valid,
            "semantic_valid": self.semantic_valid,
            "uniqueness_valid": self.uniqueness_valid,
            "referential_valid": self.referential_valid,
            "query_readiness_valid": self.query_readiness_valid,
            "interaction_ratio_per_tw": round(self.interaction_ratio_per_tw, 6),
            "multi_location_entities": self.multi_location_entities,
            "cross_partition_entities": self.cross_partition_entities,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publication-level validator for VideoGraphDB CSV datasets")
    ap.add_argument("--data_dir", required=True, help="Root dataset directory")
    ap.add_argument("--tw-seconds", type=int, default=10, help="TimeWindow size in seconds")
    ap.add_argument("--fail-fast", action="store_true", help="Stop immediately on first validation error")
    ap.add_argument("--json-out", default="", help="Optional path to write JSON validation summary")
    return ap.parse_args()


def read_csv_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_date_only(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)


def parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def parse_time_of_day(date_str: str, hhmmss: str) -> datetime:
    return datetime.fromisoformat(f"{date_str} {hhmmss}")


def tw_bucket(dt: datetime, tw_seconds: int) -> str:
    base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    bucket = int((dt - base).total_seconds()) // tw_seconds
    return f"TW{bucket:04d}"


def safe_float(x: str) -> Optional[float]:
    if x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def check_headers(root: Path, issues: IssueCollector) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for fname, expected in EXPECTED_HEADERS.items():
        path = root / fname
        if not path.exists():
            # Some datasets may omit optional files; core files must exist.
            core = fname in {"nodes_location.csv", "nodes_camera.csv", "partitions.csv", "nodes_video.csv", "nodes_person_TW.csv", "nodes_thing_TW.csv", "nodes_vehicle_TW.csv", "rels.csv"}
            if core:
                issues.error(f"Missing required file: {fname}")
            continue
        header = read_csv_header(path)
        missing = [h for h in expected if h not in header]
        extra = [h for h in header if h not in expected]
        if missing:
            issues.error(f"{fname}: missing columns {missing}")
        if extra:
            issues.warn(f"{fname}: extra columns {extra}")
        dfs[fname] = load_csv(path)
    return dfs


def assert_unique(df: pd.DataFrame, col: str, label: str, issues: IssueCollector) -> None:
    if col not in df.columns:
        issues.error(f"{label}: missing unique id column {col}")
        return
    duplicates = df[df[col].duplicated(keep=False)][col].tolist()
    if duplicates:
        sample = sorted(set(duplicates))[:10]
        issues.error(f"{label}: duplicate values in {col}, sample={sample}")


def build_node_indexes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    idx: Dict[str, Set[str]] = {}
    if "nodes_location.csv" in dfs:
        idx["loc_id"] = set(dfs["nodes_location.csv"]["loc_id"])
    if "nodes_camera.csv" in dfs:
        idx["camera_id"] = set(dfs["nodes_camera.csv"]["camera_id"])
    if "nodes_video.csv" in dfs:
        idx["video_id"] = set(dfs["nodes_video.csv"]["video_id"])
    if "nodes_person.csv" in dfs:
        idx["pid"] = set(dfs["nodes_person.csv"]["pid"])
    if "nodes_thing.csv" in dfs:
        idx["tid"] = set(dfs["nodes_thing.csv"]["tid"])
    if "nodes_vehicle.csv" in dfs:
        idx["vid"] = set(dfs["nodes_vehicle.csv"]["vid"])
    if "nodes_person_TW.csv" in dfs:
        idx["pid_tw"] = set(dfs["nodes_person_TW.csv"]["pid_tw"])
    if "nodes_thing_TW.csv" in dfs:
        idx["tid_tw"] = set(dfs["nodes_thing_TW.csv"]["tid_tw"])
    if "nodes_vehicle_TW.csv" in dfs:
        idx["vid_tw"] = set(dfs["nodes_vehicle_TW.csv"]["vid_tw"])
    return idx


def validate_uniqueness(dfs: Dict[str, pd.DataFrame], issues: IssueCollector) -> None:
    checks = [
        ("nodes_location.csv", "loc_id", "Location"),
        ("nodes_camera.csv", "camera_id", "Camera"),
        ("nodes_video.csv", "video_id", "Video"),
        ("nodes_person.csv", "pid", "Person"),
        ("nodes_thing.csv", "tid", "Thing"),
        ("nodes_vehicle.csv", "vid", "Vehicle"),
        ("nodes_person_TW.csv", "pid_tw", "Person_TW"),
        ("nodes_thing_TW.csv", "tid_tw", "Thing_TW"),
        ("nodes_vehicle_TW.csv", "vid_tw", "Vehicle_TW"),
    ]
    for fname, col, label in checks:
        if fname in dfs:
            assert_unique(dfs[fname], col, label, issues)


def validate_timewindows(dfs: Dict[str, pd.DataFrame], tw_seconds: int, issues: IssueCollector) -> Dict[Tuple[str, str], Tuple[datetime, datetime]]:
    tw_map: Dict[Tuple[str, str], Tuple[datetime, datetime]] = {}
    df = dfs.get("nodes_timewindow.csv")
    if df is None:
        issues.warn("nodes_timewindow.csv not found; skipping explicit TW row validation")
        return tw_map
    for i, row in df.iterrows():
        try:
            dt_start = parse_time_of_day(row["date"], row["start_time"])
            dt_end = parse_time_of_day(row["date"], row["end_time"])
            if dt_start >= dt_end:
                issues.error(f"nodes_timewindow.csv line {i+2}: start_time >= end_time")
                continue
            expected = tw_bucket(dt_start, tw_seconds)
            if row["tw_id"] != expected:
                issues.error(f"nodes_timewindow.csv line {i+2}: tw_id={row['tw_id']} expected {expected}")
            if row.get("duration_seconds", ""):
                try:
                    dur = int(float(row["duration_seconds"]))
                    actual = int((dt_end - dt_start).total_seconds())
                    if dur != actual:
                        issues.error(f"nodes_timewindow.csv line {i+2}: duration_seconds={dur} actual={actual}")
                except Exception:
                    issues.error(f"nodes_timewindow.csv line {i+2}: invalid duration_seconds={row['duration_seconds']}")
            tw_map[(row["date"], row["tw_id"])] = (dt_start, dt_end)
        except Exception as exc:
            issues.error(f"nodes_timewindow.csv line {i+2}: invalid datetime content ({exc})")
    return tw_map


def validate_videos(dfs: Dict[str, pd.DataFrame], issues: IssueCollector) -> Dict[str, Dict[str, object]]:
    video_map: Dict[str, Dict[str, object]] = {}
    videos = dfs.get("nodes_video.csv")
    if videos is None:
        return video_map
    cameras = dfs.get("nodes_camera.csv")
    parts = dfs.get("partitions.csv")
    camera_ids = set(cameras["camera_id"]) if cameras is not None else set()
    valid_partitions = set(parts["partition_id"]) if parts is not None else set()
    for i, row in videos.iterrows():
        try:
            dt_start = parse_time_of_day(row["date"], row["start_time"])
            dt_end = parse_time_of_day(row["date"], row["end_time"])
            if dt_start >= dt_end:
                issues.error(f"nodes_video.csv line {i+2}: start_time >= end_time for {row['video_id']}")
            if row["camera_id"] not in camera_ids:
                issues.error(f"nodes_video.csv line {i+2}: unknown camera_id {row['camera_id']}")
            if valid_partitions and row["partition_id"] not in valid_partitions:
                issues.error(f"nodes_video.csv line {i+2}: unknown partition_id {row['partition_id']}")
            video_map[row["video_id"]] = {
                "camera_id": row["camera_id"],
                "partition_id": row["partition_id"],
                "date": row["date"],
                "start": dt_start,
                "end": dt_end,
            }
        except Exception as exc:
            issues.error(f"nodes_video.csv line {i+2}: invalid video datetime content ({exc})")
    return video_map


def validate_dynamic_nodes(dfs: Dict[str, pd.DataFrame], issues: IssueCollector) -> Dict[str, Dict[str, Dict[str, str]]]:
    result: Dict[str, Dict[str, Dict[str, str]]] = {"person": {}, "thing": {}, "vehicle": {}}
    mappings = [
        ("nodes_person_TW.csv", "pid_tw", "pid", "person"),
        ("nodes_thing_TW.csv", "tid_tw", "tid", "thing"),
        ("nodes_vehicle_TW.csv", "vid_tw", "vid", "vehicle"),
    ]
    for fname, tw_col, base_col, kind in mappings:
        df = dfs.get(fname)
        base_df_name = f"nodes_{kind}.csv"
        base_df = dfs.get(base_df_name)
        base_ids = set(base_df[base_col]) if base_df is not None else set()
        if df is None:
            continue
        for i, row in df.iterrows():
            if row["id_global"] not in base_ids and base_ids:
                issues.error(f"{fname} line {i+2}: id_global {row['id_global']} not found in {base_df_name}")
            result[kind][row[tw_col]] = {
                "id_global": row["id_global"],
                "date": row["date"],
                "tw_id": row["tw_id"],
                "partition_id": row["partition_id"],
            }
    return result


def build_camera_location_map(rels: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    loc_df = rels[rels["type"] == "LOCATED_AT"]
    for _, row in loc_df.iterrows():
        mapping[row["source_id"]] = row["destination_id"]
    return mapping


def validate_edges(
    dfs: Dict[str, pd.DataFrame],
    video_map: Dict[str, Dict[str, object]],
    tw_map: Dict[Tuple[str, str], Tuple[datetime, datetime]],
    dynamic_nodes: Dict[str, Dict[str, Dict[str, str]]],
    tw_seconds: int,
    issues: IssueCollector,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    rels = dfs.get("rels.csv")
    if rels is None:
        issues.error("rels.csv is required")
        return pd.DataFrame(), {}

    node_ids = build_node_indexes(dfs)
    camera_to_location = build_camera_location_map(rels)
    detection_info: Dict[str, Dict[str, object]] = {}
    detection_counts: Counter[str] = Counter()

    # Build location type map for semantic checks
    loc_type_map = {}
    if "nodes_location.csv" in dfs:
        loc_type_map = dict(zip(dfs["nodes_location.csv"]["loc_id"], dfs["nodes_location.csv"]["loc_type"]))

    # Track undirected checks later
    interacts_pairs: Counter[Tuple[str, str, str, str, str, str]] = Counter()
    nearby_pairs: Counter[Tuple[str, str]] = Counter()

    for i, row in rels.iterrows():
        line_no = i + 2
        etype = row["type"]
        if etype not in ALL_EDGE_TYPES:
            issues.error(f"rels.csv line {line_no}: unknown edge type {etype}")
            continue

        src = row["source_id"]
        dst = row["destination_id"]
        date = row["date"]
        tw_id = row["tw_id"]
        part = row["partition_id"]
        camera_id = row["camera_id"]
        location_id = row["location_id"]
        conf = row["confidence"]
        bbox = row["bbox"]

        # Field-shape validation by edge type
        if etype == "LOCATED_AT":
            if src not in node_ids.get("camera_id", set()):
                issues.error(f"rels.csv line {line_no}: LOCATED_AT source camera {src} not found")
            if dst not in node_ids.get("loc_id", set()):
                issues.error(f"rels.csv line {line_no}: LOCATED_AT destination location {dst} not found")
            if row["ts_start"] or row["ts_end"] or date or tw_id or conf or bbox:
                issues.error(f"rels.csv line {line_no}: LOCATED_AT should not contain temporal/confidence/bbox fields")
            if camera_id and camera_id != src:
                issues.error(f"rels.csv line {line_no}: LOCATED_AT camera_id must match source_id")
            if location_id and location_id != dst:
                issues.error(f"rels.csv line {line_no}: LOCATED_AT location_id must match destination_id")
            continue

        if etype == "NEAR_BY":
            if src not in node_ids.get("loc_id", set()) or dst not in node_ids.get("loc_id", set()):
                issues.error(f"rels.csv line {line_no}: NEAR_BY references unknown location(s)")
            if src == dst:
                issues.error(f"rels.csv line {line_no}: NEAR_BY self-loop is not allowed")
            if any([row["ts_start"], row["ts_end"], date, tw_id, camera_id, location_id, conf, bbox]):
                issues.error(f"rels.csv line {line_no}: NEAR_BY should not contain temporal/context/confidence/bbox fields")
            nearby_pairs[(src, dst)] += 1
            continue

        # Parse timestamps for remaining edges where they are expected/allowed
        ts_start_raw = row["ts_start"]
        ts_end_raw = row["ts_end"]
        dt_start: Optional[datetime] = None
        dt_end: Optional[datetime] = None
        if ts_start_raw:
            try:
                dt_start = parse_ts(ts_start_raw)
            except Exception as exc:
                issues.error(f"rels.csv line {line_no}: invalid ts_start ({exc})")
        if ts_end_raw:
            try:
                dt_end = parse_ts(ts_end_raw)
            except Exception as exc:
                issues.error(f"rels.csv line {line_no}: invalid ts_end ({exc})")

        if etype == "RECORDED_BY":
            if src not in node_ids.get("video_id", set()):
                issues.error(f"rels.csv line {line_no}: RECORDED_BY source video {src} not found")
            if dst not in node_ids.get("camera_id", set()):
                issues.error(f"rels.csv line {line_no}: RECORDED_BY destination camera {dst} not found")
            if not (dt_start and dt_end and date):
                issues.error(f"rels.csv line {line_no}: RECORDED_BY requires ts_start, ts_end, date")
            elif dt_start >= dt_end:
                issues.error(f"rels.csv line {line_no}: RECORDED_BY requires ts_start < ts_end")
            if tw_id:
                issues.error(f"rels.csv line {line_no}: RECORDED_BY tw_id must be blank")
            # Compare against nodes_video if available
            video = video_map.get(src)
            if video is not None:
                if dst != video["camera_id"]:
                    issues.error(f"rels.csv line {line_no}: RECORDED_BY camera mismatch with nodes_video for {src}")
                if dt_start and dt_start != video["start"]:
                    issues.error(f"rels.csv line {line_no}: RECORDED_BY ts_start mismatch with nodes_video for {src}")
                if dt_end and dt_end != video["end"]:
                    issues.error(f"rels.csv line {line_no}: RECORDED_BY ts_end mismatch with nodes_video for {src}")
            continue

        if etype == "NEXT_TO":
            if src not in node_ids.get("video_id", set()) or dst not in node_ids.get("video_id", set()):
                issues.error(f"rels.csv line {line_no}: NEXT_TO references unknown video(s)")
            # Allow empty timestamps or equal timestamps depending on generator choice
            if dt_start and dt_end and dt_start != dt_end:
                issues.error(f"rels.csv line {line_no}: NEXT_TO requires ts_start == ts_end when timestamps are present")
            continue

        # Temporal/semantic edges from here onward
        if etype == "DETECTED_IN":
            if src not in (node_ids.get("pid_tw", set()) | node_ids.get("tid_tw", set()) | node_ids.get("vid_tw", set())):
                issues.error(f"rels.csv line {line_no}: DETECTED_IN source entity_tw {src} not found")
            if dst not in node_ids.get("video_id", set()):
                issues.error(f"rels.csv line {line_no}: DETECTED_IN destination video {dst} not found")
            if not (dt_start and dt_end and date and tw_id):
                issues.error(f"rels.csv line {line_no}: DETECTED_IN requires ts_start, ts_end, date, tw_id")
                continue
            if not dt_start < dt_end:
                issues.error(f"rels.csv line {line_no}: DETECTED_IN requires ts_start < ts_end")
            expected_tw = tw_bucket(dt_start, tw_seconds)
            if tw_id != expected_tw:
                issues.error(f"rels.csv line {line_no}: DETECTED_IN tw_id={tw_id} expected {expected_tw}")
            tw_interval = tw_map.get((date, tw_id))
            if tw_interval:
                tw_start, tw_end = tw_interval
                if not (tw_start <= dt_start < dt_end <= tw_end):
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN not within TW interval [{tw_start}, {tw_end})")
            video = video_map.get(dst)
            if video:
                if not (video["start"] <= dt_start < dt_end <= video["end"]):
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN not within video interval for {dst}")
                if camera_id and camera_id != video["camera_id"]:
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN camera_id mismatch with video.camera_id")
                expected_loc = camera_to_location.get(video["camera_id"])
                if expected_loc and location_id and location_id != expected_loc:
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN location_id mismatch with camera LOCATED_AT")
                if part and video["partition_id"] and part != video["partition_id"]:
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN partition_id mismatch with video.partition_id")
            # Exactly one detection per Entity_TW
            detection_counts[src] += 1
            detection_info[src] = {
                "ts_start": dt_start,
                "ts_end": dt_end,
                "video_id": dst,
                "camera_id": camera_id or (video["camera_id"] if video else ""),
                "location_id": location_id or camera_to_location.get(camera_id or (video["camera_id"] if video else ""), ""),
                "date": date,
                "tw_id": tw_id,
                "partition_id": part,
            }
            # Optional confidence/bbox validation
            if conf:
                v = safe_float(conf)
                if v is None or not (0.0 <= v <= 1.0):
                    issues.error(f"rels.csv line {line_no}: DETECTED_IN invalid confidence={conf}")
            if bbox:
                try:
                    json.loads(bbox)
                except Exception as exc:
                    issues.error(f"rels.csv line {line_no}: invalid bbox JSON ({exc})")
            continue

        # Semantic edges
        if etype == "CARRIES":
            if src not in node_ids.get("pid_tw", set()):
                issues.error(f"rels.csv line {line_no}: CARRIES source {src} is not Person_TW")
            if dst not in node_ids.get("tid_tw", set()):
                issues.error(f"rels.csv line {line_no}: CARRIES destination {dst} is not Thing_TW")
        elif etype == "USES":
            if src not in node_ids.get("pid_tw", set()):
                issues.error(f"rels.csv line {line_no}: USES source {src} is not Person_TW")
            if dst not in node_ids.get("vid_tw", set()):
                issues.error(f"rels.csv line {line_no}: USES destination {dst} is not Vehicle_TW")
        elif etype == "INTERACTS_WITH":
            if src not in node_ids.get("pid_tw", set()) or dst not in node_ids.get("pid_tw", set()):
                issues.error(f"rels.csv line {line_no}: INTERACTS_WITH must connect Person_TW to Person_TW")
            if src == dst:
                issues.error(f"rels.csv line {line_no}: INTERACTS_WITH self-edge is not allowed")

        if not (dt_start and dt_end and date and tw_id):
            issues.error(f"rels.csv line {line_no}: {etype} requires ts_start, ts_end, date, tw_id")
            continue
        if not dt_start < dt_end:
            issues.error(f"rels.csv line {line_no}: {etype} requires ts_start < ts_end")
            continue
        expected_tw = tw_bucket(dt_start, tw_seconds)
        if tw_id != expected_tw:
            issues.error(f"rels.csv line {line_no}: {etype} tw_id={tw_id} expected {expected_tw}")
        tw_interval = tw_map.get((date, tw_id))
        if tw_interval:
            tw_start, tw_end = tw_interval
            if not (tw_start <= dt_start < dt_end <= tw_end):
                issues.error(f"rels.csv line {line_no}: {etype} not within TW interval [{tw_start}, {tw_end})")

        src_det = detection_info.get(src)
        dst_det = detection_info.get(dst)
        if src_det is None or dst_det is None:
            issues.error(f"rels.csv line {line_no}: {etype} references entity without DETECTED_IN")
            continue
        if src_det["tw_id"] != tw_id or dst_det["tw_id"] != tw_id:
            issues.error(f"rels.csv line {line_no}: {etype} participants must share same TW")
        if src_det["video_id"] != dst_det["video_id"]:
            issues.error(f"rels.csv line {line_no}: {etype} participants must be in same video")
        overlap_start = max(src_det["ts_start"], dst_det["ts_start"])
        overlap_end = min(src_det["ts_end"], dst_det["ts_end"])
        if not overlap_start < overlap_end:
            issues.error(f"rels.csv line {line_no}: {etype} participants have no positive overlap")
        elif not (overlap_start <= dt_start < dt_end <= overlap_end):
            issues.error(f"rels.csv line {line_no}: {etype} interval must be within detection intersection")
        # Optional confidence field validation
        if conf:
            v = safe_float(conf)
            if v is None or not (0.0 <= v <= 1.0):
                issues.error(f"rels.csv line {line_no}: {etype} invalid confidence={conf}")

        if etype == "USES":
            loc_type = loc_type_map.get(src_det["location_id"], "")
            if loc_type in {"Indoor", "Office"}:
                issues.error(f"rels.csv line {line_no}: USES occurs in invalid location type {loc_type}")
        elif etype == "INTERACTS_WITH":
            key = (src, dst, date, tw_id, dt_start.isoformat(sep=" "), dt_end.isoformat(sep=" "))
            interacts_pairs[key] += 1

    # Exactly one DETECTED_IN per Entity_TW
    all_entity_tw = node_ids.get("pid_tw", set()) | node_ids.get("tid_tw", set()) | node_ids.get("vid_tw", set())
    for ent_id in sorted(all_entity_tw):
        n = detection_counts.get(ent_id, 0)
        if n != 1:
            issues.error(f"Entity_TW {ent_id} must have exactly 1 DETECTED_IN, found {n}")

    # Undirected consistency checks for INTERACTS_WITH and NEAR_BY
    for (src, dst), count in list(nearby_pairs.items()):
        if nearby_pairs.get((dst, src), 0) == 0:
            issues.error(f"Missing symmetric NEAR_BY edge: {dst} -> {src}")
    # For interacts, require exact reverse edge with same interval/date/tw_id
    for (src, dst, date, tw_id, s, e), count in list(interacts_pairs.items()):
        if interacts_pairs.get((dst, src, date, tw_id, s, e), 0) == 0:
            issues.error(f"Missing symmetric INTERACTS_WITH edge: {dst} -> {src} for interval [{s}, {e})")

    return rels, detection_info


def validate_duplicate_edges(rels: pd.DataFrame, issues: IssueCollector) -> None:
    dup_cols = ["source_id", "destination_id", "type", "ts_start", "ts_end"]
    dups = rels[rels.duplicated(subset=dup_cols, keep=False)]
    if not dups.empty:
        sample = dups[dup_cols].drop_duplicates().head(10).to_dict(orient="records")
        issues.error(f"Duplicate edges detected, sample={sample}")


def validate_partition_integrity(root: Path, issues: IssueCollector) -> Tuple[int, int]:
    by_partition = root / "by_partition"
    if not by_partition.exists():
        issues.warn("by_partition directory not found; skipping partition row-count checks")
        return (0, 0)

    total_root_rows = 0
    total_partition_rows = 0
    for fname in ROOT_FILES_TO_PARTITION_CHECK:
        root_file = root / fname
        if not root_file.exists():
            continue
        with root_file.open("r", newline="", encoding="utf-8") as f:
            root_rows = max(sum(1 for _ in f) - 1, 0)
        part_rows = 0
        for partition_dir in by_partition.iterdir():
            if not partition_dir.is_dir():
                continue
            part_file = partition_dir / fname
            if part_file.exists():
                with part_file.open("r", newline="", encoding="utf-8") as pf:
                    part_rows += max(sum(1 for _ in pf) - 1, 0)
        total_root_rows += root_rows
        total_partition_rows += part_rows
        # Static files are replicated by design in current generator, skip strict equality there.
        if fname in {"nodes_location.csv", "nodes_camera.csv", "partitions.csv", "nodes_timewindow.csv", "nodes_video.csv"}:
            continue
        if part_rows != root_rows:
            issues.error(f"Partition mismatch for {fname}: root_rows={root_rows}, partition_rows={part_rows}")
    return total_root_rows, total_partition_rows


def validate_semantics(dfs: Dict[str, pd.DataFrame], detection_info: Dict[str, Dict[str, object]], issues: IssueCollector) -> Tuple[int, int]:
    locations = dfs.get("nodes_location.csv")
    if locations is None:
        return (0, 0)
    loc_type_map = dict(zip(locations["loc_id"], locations["loc_type"]))

    person_tw = dfs.get("nodes_person_TW.csv", pd.DataFrame())
    vehicle_tw = dfs.get("nodes_vehicle_TW.csv", pd.DataFrame())

    # Vehicle semantic constraint: must not appear in Indoor / Office.
    bad_vehicle = 0
    for vid_tw in vehicle_tw.get("vid_tw", []):
        det = detection_info.get(vid_tw)
        if not det:
            continue
        loc_type = loc_type_map.get(det["location_id"], "")
        if loc_type in {"Indoor", "Office"}:
            bad_vehicle += 1
            issues.error(f"Vehicle_TW {vid_tw} detected in invalid location type {loc_type}")

    # Query readiness signals: number of entities moving across locations/partitions.
    movement_df = person_tw.copy()
    if not movement_df.empty and "pid_tw" in movement_df.columns:
        movement_df["location_id"] = movement_df["pid_tw"].map(lambda x: detection_info.get(x, {}).get("location_id", ""))
        movement_df["partition_id_det"] = movement_df["pid_tw"].map(lambda x: detection_info.get(x, {}).get("partition_id", ""))
        multi_loc = int((movement_df.groupby("id_global")["location_id"].nunique() > 1).sum())
        cross_part = int((movement_df.groupby("id_global")["partition_id_det"].nunique() > 1).sum())
    else:
        multi_loc, cross_part = 0, 0
    return multi_loc, cross_part


def compute_interaction_ratio_per_tw(rels: pd.DataFrame, person_tw: pd.DataFrame) -> float:
    if rels.empty or person_tw.empty:
        return 0.0
    if not {"date", "tw_id"}.issubset(person_tw.columns):
        return 0.0
    person_counts = person_tw.groupby(["date", "tw_id"]).size()
    denom = float(sum(n * (n - 1) / 2.0 for n in person_counts if n > 1))
    if denom == 0:
        return 0.0
    interacts = rels[rels["type"] == "INTERACTS_WITH"]
    if interacts.empty:
        return 0.0
    # Divide by 2 because undirected semantics are stored as two directed edges.
    num = float(len(interacts) / 2.0)
    return min(num / denom, 1.0)


def validate_query_readiness(dfs: Dict[str, pd.DataFrame], rels: pd.DataFrame, issues: IssueCollector) -> None:
    # Basic existence checks for path / sequence / interaction / heterogeneous pattern support.
    needed = {"DETECTED_IN", "RECORDED_BY", "LOCATED_AT", "INTERACTS_WITH", "CARRIES", "USES", "NEAR_BY"}
    actual = set(rels["type"].unique()) if not rels.empty else set()
    missing = needed - actual
    if missing:
        issues.error(f"Query-readiness failure: missing required edge types {sorted(missing)}")


def verify_dataset(data_dir: str, tw_seconds: int = 10, fail_fast: bool = False) -> Tuple[ValidationSummary, IssueCollector]:
    root = Path(data_dir)
    issues = IssueCollector(fail_fast=fail_fast)
    summary = ValidationSummary()

    dfs = check_headers(root, issues)
    validate_uniqueness(dfs, issues)
    node_indexes = build_node_indexes(dfs)
    tw_map = validate_timewindows(dfs, tw_seconds, issues)
    video_map = validate_videos(dfs, issues)
    dynamic_nodes = validate_dynamic_nodes(dfs, issues)
    rels, detection_info = validate_edges(dfs, video_map, tw_map, dynamic_nodes, tw_seconds, issues)
    if not rels.empty:
        validate_duplicate_edges(rels, issues)
        validate_query_readiness(dfs, rels, issues)
    total_root_rows, total_partition_rows = validate_partition_integrity(root, issues)
    multi_loc, cross_part = validate_semantics(dfs, detection_info, issues)

    # Summary status mapping
    if issues.errors:
        # Derive coarse statuses from message categories.
        text = "\n".join(issues.errors)
        if any(k in text for k in ["duplicate values", "Duplicate edges"]):
            summary.uniqueness_valid = "FAIL"
        if any(k in text for k in ["not found", "references unknown", "Missing required file"]):
            summary.referential_valid = "FAIL"
        if any(k in text for k in ["tw_id=", "start_time >= end_time", "invalid ts_", "requires ts_start < ts_end", "within TW interval", "within video interval"]):
            summary.detection_valid = "FAIL"
            summary.schema_valid = "FAIL"
        if any(k in text for k in ["INTERACTS_WITH", "CARRIES", "USES", "NEAR_BY", "positive overlap", "symmetric"]):
            summary.relation_valid = "FAIL"
        if any(k in text for k in ["invalid location type", "Vehicle_TW"]):
            summary.semantic_valid = "FAIL"
        if any(k in text for k in ["Partition mismatch", "by_partition"]):
            summary.partition_valid = "FAIL"
        if any(k in text for k in ["Query-readiness failure"]):
            summary.query_readiness_valid = "FAIL"
        if any(k in text for k in ["Entity_TW", "DETECTED_IN"]):
            summary.entity_valid = "FAIL"
            summary.detection_valid = "FAIL"

    if rels is not None and not rels.empty and "nodes_person_TW.csv" in dfs:
        summary.interaction_ratio_per_tw = compute_interaction_ratio_per_tw(rels, dfs["nodes_person_TW.csv"])
    summary.multi_location_entities = multi_loc
    summary.cross_partition_entities = cross_part
    summary.total_errors = len(issues.errors)
    summary.total_warnings = len(issues.warnings)

    # If no errors, keep PASS everywhere.
    if not issues.errors:
        summary = ValidationSummary(
            interaction_ratio_per_tw=summary.interaction_ratio_per_tw,
            multi_location_entities=summary.multi_location_entities,
            cross_partition_entities=summary.cross_partition_entities,
            total_errors=0,
            total_warnings=len(issues.warnings),
        )
    return summary, issues


def main() -> None:
    args = parse_args()
    try:
        summary, issues = verify_dataset(args.data_dir, tw_seconds=args.tw_seconds, fail_fast=args.fail_fast)
    except ValidationError as exc:
        # Fail-fast mode: print the first error and exit.
        print(f"[FAIL-FAST] {exc}")
        sys.exit(1)

    print(f"Validating dataset at: {args.data_dir}\n")
    if issues.warnings:
        print("Warnings:")
        for msg in issues.warnings:
            print(f"  - {msg}")
        print()
    if issues.errors:
        print("Errors:")
        for msg in issues.errors:
            print(f"  - {msg}")
        print()

    print("Validation summary:")
    rendered = json.dumps(summary.to_dict(), indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(rendered + "\n", encoding="utf-8")

    sys.exit(0 if not issues.errors else 1)


if __name__ == "__main__":
    main()
