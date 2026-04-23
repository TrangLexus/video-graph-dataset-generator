#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

TEMPORAL_EDGE_TYPES = {"DETECTED_IN", "CARRIES", "USES", "INTERACTS_WITH"}
STRUCTURAL_EDGE_TYPES = {"LOCATED_AT", "RECORDED_BY", "NEAR_BY", "NEXT_TO"}
ALL_EDGE_TYPES = TEMPORAL_EDGE_TYPES | STRUCTURAL_EDGE_TYPES

REQUIRED_ROOT_FILES = [
    "nodes_location.csv",
    "nodes_camera.csv",
    "partitions.csv",
    "nodes_person.csv",
    "nodes_thing.csv",
    "nodes_vehicle.csv",
    "nodes_timewindow.csv",
    "nodes_video.csv",
    "nodes_person_TW.csv",
    "nodes_thing_TW.csv",
    "nodes_vehicle_TW.csv",
    "rels.csv",
]

REQUIRED_COLUMNS = {
    "nodes_location.csv": {"location_id", "name", "location_type"},
    "nodes_camera.csv": {"camera_id", "name", "view_type", "is_indoor"},
    "partitions.csv": {"partition_id", "location_id"},
    "nodes_person.csv": {"person_id", "gender", "age_group"},
    "nodes_thing.csv": {"thing_id", "thing_type", "size_category", "base_color"},
    "nodes_vehicle.csv": {"vehicle_id", "vehicle_type", "base_color"},
    "nodes_timewindow.csv": {"date", "tw_id", "start_time", "end_time", "duration_seconds"},
    "nodes_video.csv": {
        "video_id", "camera_id", "partition_id", "date",
        "start_time", "end_time", "fps", "resolution"
    },
    "nodes_person_TW.csv": {
        "pid_tw", "person_id", "date", "tw_id", "partition_id",
        "shirt_color", "pant_color", "pose_state"
    },
    "nodes_thing_TW.csv": {
        "tid_tw", "thing_id", "date", "tw_id", "partition_id",
        "thing_type", "size_category", "base_color", "state"
    },
    "nodes_vehicle_TW.csv": {
        "vid_tw", "vehicle_id", "date", "tw_id", "partition_id",
        "vehicle_type", "base_color", "speed_kmh", "direction"
    },
    "rels.csv": {
        "source_id", "destination_id", "type", "start_time", "end_time", "date",
        "tw_id", "partition_id", "camera_id", "location_id",
        "confidence", "bbox", "description"
    },
}

ENTITY_NODE_FILES = {
    "nodes_person_TW.csv": ("pid_tw", "person_id"),
    "nodes_thing_TW.csv": ("tid_tw", "thing_id"),
    "nodes_vehicle_TW.csv": ("vid_tw", "vehicle_id"),
}

STORAGE_PARTITION_AGG_FILES = [
    "nodes_video.csv",
    "nodes_person_TW.csv",
    "nodes_thing_TW.csv",
    "nodes_vehicle_TW.csv",
    "rels.csv",
]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class TimeWindowInfo:
    date: str
    tw_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    camera_id: str
    partition_id: str
    date: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class DetectionInfo:
    entity_tw_id: str
    start_time: datetime
    end_time: datetime
    video_id: str
    camera_id: str
    location_id: str
    date: str
    tw_id: str
    partition_id: str


# =============================================================================
# ISSUE COLLECTOR
# =============================================================================

@dataclass
class IssueCollector:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    categories: Set[str] = field(default_factory=set)

    def error(self, msg: str, category: str = "general") -> None:
        self.errors.append(msg)
        self.categories.add(category)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# =============================================================================
# STATS / SUMMARY
# =============================================================================

@dataclass
class DatasetStats:
    file_row_counts: Dict[str, int] = field(default_factory=dict)
    root_files_present: List[str] = field(default_factory=list)
    root_files_missing: List[str] = field(default_factory=list)

    global_entity_counts: Dict[str, int] = field(default_factory=dict)
    temporal_entity_counts: Dict[str, int] = field(default_factory=dict)

    edge_type_counts: Dict[str, int] = field(default_factory=dict)
    edge_type_ratios_vs_detected_in: Dict[str, float] = field(default_factory=dict)

    detections_total: int = 0
    detections_person: int = 0
    detections_thing: int = 0
    detections_vehicle: int = 0

    person_vehicle_same_context_pairs: int = 0
    person_thing_same_context_pairs: int = 0
    person_person_same_context_pairs: int = 0

    semantic_edge_same_context_ok: int = 0
    semantic_edge_overlap_ok: int = 0

    by_partition_dirs: int = 0
    by_partition_expected_dirs: int = 0
    by_partition_rel_rows_total: int = 0
    by_partition_rel_rows_match_root: bool = False

    date_anchor_dirs: int = 0
    date_anchor_expected_dirs: int = 0
    date_anchor_dates_expected: List[str] = field(default_factory=list)
    date_anchor_dates_found: List[str] = field(default_factory=list)
    date_anchor_loc_dirs: int = 0


@dataclass
class ValidationSummary:
    relation_valid: str = "PASS"
    detection_valid: str = "PASS"
    semantic_valid: str = "PASS"
    schema_valid: str = "PASS"
    file_valid: str = "PASS"
    storage_valid: str = "PASS"

    total_errors: int = 0
    total_warnings: int = 0


# =============================================================================
# UTILS
# =============================================================================

def parse_ts(x: str) -> datetime:
    return datetime.fromisoformat(x)


def parse_time(date: str, t: str) -> datetime:
    return datetime.fromisoformat(f"{date} {t}")


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _fmt_ratio(x: float) -> str:
    return f"{x:.6f}"


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = sorted(df.columns.tolist())
    out = df[cols].copy()
    for c in cols:
        out[c] = out[c].astype(str)
    return out.sort_values(by=cols, kind="mergesort").reset_index(drop=True)


def df_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "EMPTY"
    norm = normalize_df(df)
    payload = norm.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def concat_csvs(paths: List[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [read_csv_safe(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def collect_partition_files(root: Path, filename: str) -> List[Path]:
    part_root = root / "by_partition"
    if not part_root.exists():
        return []
    return sorted([p for p in part_root.glob("partition_*/" + filename) if p.is_file()])


def collect_date_files(root: Path, filename: str) -> List[Path]:
    return sorted([p for p in root.glob("date=*/loc=*/" + filename) if p.is_file()])


def extract_date_from_dir(path: Path) -> str:
    for part in path.parts:
        if part.startswith("date="):
            return part.split("=", 1)[1]
    return ""


def extract_loc_from_dir(path: Path) -> str:
    for part in path.parts:
        if part.startswith("loc="):
            return part.split("=", 1)[1]
    return ""


# =============================================================================
# FILE / SCHEMA VALIDATION
# =============================================================================

def load_root_csvs(root: Path, issues: IssueCollector, stats: DatasetStats) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for fname in REQUIRED_ROOT_FILES:
        path = root / fname
        if path.exists():
            stats.root_files_present.append(fname)
            try:
                dfs[fname] = read_csv_safe(path)
                stats.file_row_counts[fname] = len(dfs[fname])
            except Exception as e:
                issues.error(f"Failed to read {fname}: {e}", "file")
        else:
            stats.root_files_missing.append(fname)
            issues.error(f"Missing required file: {fname}", "file")
    return dfs


def validate_required_columns(dfs: Dict[str, pd.DataFrame], issues: IssueCollector) -> None:
    for fname, expected_cols in REQUIRED_COLUMNS.items():
        if fname not in dfs:
            continue
        actual_cols = set(dfs[fname].columns.tolist())
        missing = expected_cols - actual_cols
        if missing:
            issues.error(f"{fname} missing columns: {sorted(missing)}", "schema")

def build_location_type_map(dfs: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    if "nodes_location.csv" not in dfs:
        return {}
    df = dfs["nodes_location.csv"]
    return {
        str(r.location_id): str(r.location_type)
        for r in df.itertuples(index=False)
    }


def build_camera_location_map(rels: pd.DataFrame) -> Dict[str, str]:
    camera_to_location: Dict[str, str] = {}
    for i, row in rels.iterrows():
        if str(row["type"]) != "LOCATED_AT":
            continue
        cam_id = str(row["source_id"])
        loc_id = str(row["destination_id"])
        prev = camera_to_location.get(cam_id)
        if prev is not None and prev != loc_id:
            # để validator khác bắt lỗi nhiều-camera-location nếu cần
            continue
        camera_to_location[cam_id] = loc_id
    return camera_to_location


# =============================================================================
# CORE DATASET VALIDATION
# =============================================================================

def validate_timewindows(df: pd.DataFrame, issues: IssueCollector) -> Dict[Tuple[str, str], TimeWindowInfo]:
    tw_map: Dict[Tuple[str, str], TimeWindowInfo] = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        line = i + 1
        try:
            date = str(row.date)
            tw_id = str(row.tw_id)
            start = parse_time(date, str(row.start_time))
            end = parse_time(date, str(row.end_time))

            if start >= end:
                issues.error(f"TimeWindow invalid interval at line {line}: start >= end", "schema")

            key = (date, tw_id)
            if key in tw_map:
                issues.error(f"Duplicate TimeWindow key at line {line}: {key}", "schema")

            tw_map[key] = TimeWindowInfo(date=date, tw_id=tw_id, start=start, end=end)
        except Exception as e:
            issues.error(f"TimeWindow parse error at line {line}: {e}", "schema")

    return tw_map


def validate_videos(df: pd.DataFrame, issues: IssueCollector) -> Dict[str, VideoInfo]:
    videos: Dict[str, VideoInfo] = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        line = i + 1
        try:
            vid = str(row.video_id)
            date = str(row.date)
            start = parse_time(date, str(row.start_time))
            end = parse_time(date, str(row.end_time))

            if start >= end:
                issues.error(f"Video invalid interval at line {line}: start >= end", "schema")
            if vid in videos:
                issues.error(f"Duplicate video_id at line {line}: {vid}", "schema")

            videos[vid] = VideoInfo(
                video_id=vid,
                camera_id=str(row.camera_id),
                partition_id=str(row.partition_id),
                date=date,
                start=start,
                end=end,
            )
        except Exception as e:
            issues.error(f"Video parse error at line {line}: {e}", "schema")

    return videos


# def validate_edges(
#     rels: pd.DataFrame,
#     video_map: Dict[str, VideoInfo],
#     tw_map: Dict[Tuple[str, str], TimeWindowInfo],
#     issues: IssueCollector,
#     stats: DatasetStats,
# ) -> Dict[str, DetectionInfo]:
def validate_edges(
    rels: pd.DataFrame,
    video_map: Dict[str, VideoInfo],
    tw_map: Dict[Tuple[str, str], TimeWindowInfo],
    camera_to_location: Dict[str, str],
    issues: IssueCollector,
    stats: DatasetStats,
) -> Dict[str, DetectionInfo]:
    detection_info: Dict[str, DetectionInfo] = {}
    edge_type_counter = Counter(str(x) for x in rels["type"].tolist())
    stats.edge_type_counts = dict(edge_type_counter)

    for i, row in rels.iterrows():
        line = i + 2
        etype = str(row["type"])
        src = str(row["source_id"])
        dst = str(row["destination_id"])

        if etype not in ALL_EDGE_TYPES:
            issues.error(f"Unknown edge type at line {line}: {etype}", "schema")
            continue

        start_time = row["start_time"]
        end_time = row["end_time"]
        dt_start = parse_ts(start_time) if start_time else None
        dt_end = parse_ts(end_time) if end_time else None

        if etype == "NEXT_TO":
            if not (dt_start and dt_end):
                issues.error(f"NEXT_TO missing time at line {line}", "relation")
            elif dt_start != dt_end:
                issues.error(f"NEXT_TO must be point event at line {line}", "relation")
            continue

        if etype == "LOCATED_AT":
            if start_time or end_time:
                issues.warn(f"LOCATED_AT has timestamps at line {line}; expected empty")
            continue

        if etype == "NEAR_BY":
            if start_time or end_time:
                issues.warn(f"NEAR_BY has timestamps at line {line}; expected empty")
            continue

        # if etype == "RECORDED_BY":
        #     if not (dt_start and dt_end):
        #         issues.error(f"RECORDED_BY missing time at line {line}", "relation")
        #     continue

        if etype == "RECORDED_BY":
            if not (dt_start and dt_end):
                issues.error(f"RECORDED_BY missing time at line {line}", "relation")
                continue

            video = video_map.get(src)
            if not video:
                issues.error(f"RECORDED_BY references unknown video at line {line}: {src}", "relation")
                continue

            row_camera_id = str(row["camera_id"])
            row_location_id = str(row["location_id"])

            if str(dst) != video.camera_id:
                issues.error(
                    f"RECORDED_BY destination mismatch at line {line}: dst={dst}, video.camera_id={video.camera_id}",
                    "relation",
                )

            if row_camera_id and row_camera_id != str(dst):
                issues.error(
                    f"RECORDED_BY row.camera_id mismatch at line {line}: row.camera_id={row_camera_id}, dst={dst}",
                    "relation",
                )

            mapped_loc = camera_to_location.get(str(dst))
            if mapped_loc is not None and row_location_id and row_location_id != mapped_loc:
                issues.error(
                    f"RECORDED_BY row.location_id mismatch at line {line}: row.location_id={row_location_id}, "
                    f"camera->{mapped_loc}, camera={dst}",
                    "relation",
                )

            if not (video.start == dt_start and video.end == dt_end):
                issues.error(
                    f"RECORDED_BY interval mismatch at line {line}: rel=[{dt_start},{dt_end}] "
                    f"vs video=[{video.start},{video.end}] for video={src}",
                    "relation",
                )
            continue

        if etype == "DETECTED_IN":
            if not (dt_start and dt_end):
                issues.error(f"DETECTED_IN missing time at line {line}", "detection")
                continue

            date = str(row["date"])
            tw_id = str(row["tw_id"])
            tw = tw_map.get((date, tw_id))
            if not tw:
                issues.error(f"DETECTED_IN references missing TW at line {line}: {(date, tw_id)}", "detection")
                continue
            if not (tw.start <= dt_start < dt_end <= tw.end):
                issues.error(f"DETECTED_IN outside TW bounds at line {line}: {src}", "detection")

            video = video_map.get(dst)
            if not video:
                issues.error(f"DETECTED_IN references unknown video at line {line}: {dst}", "detection")
                continue
            if not (video.start <= dt_start < dt_end <= video.end):
                issues.error(f"DETECTED_IN outside video bounds at line {line}: {src}", "detection")

            row_camera_id = str(row["camera_id"])
            row_location_id = str(row["location_id"])

            if row_camera_id != video.camera_id:
                issues.error(
                    f"DETECTED_IN camera_id mismatch at line {line}: row.camera_id={row_camera_id}, "
                    f"video.camera_id={video.camera_id}, entity={src}, video={dst}",
                    "detection",
                )

            mapped_loc = camera_to_location.get(video.camera_id)
            if mapped_loc is not None and row_location_id != mapped_loc:
                issues.error(
                    f"DETECTED_IN location_id mismatch at line {line}: row.location_id={row_location_id}, "
                    f"camera->{mapped_loc}, entity={src}, camera={video.camera_id}",
                    "detection",
                )

            if src in detection_info:
                issues.error(f"Entity_TW has multiple DETECTED_IN edges: {src}", "detection")

            detection_info[src] = DetectionInfo(
                entity_tw_id=src,
                start_time=dt_start,
                end_time=dt_end,
                video_id=dst,
                camera_id=str(row["camera_id"]),
                location_id=str(row["location_id"]),
                date=date,
                tw_id=tw_id,
                partition_id=str(row["partition_id"]),
            )
            continue

        if etype in {"CARRIES", "USES", "INTERACTS_WITH"}:
            if not (dt_start and dt_end):
                issues.error(f"{etype} missing time at line {line}", "relation")

    stats.detections_total = len(detection_info)
    stats.detections_person = sum(1 for k in detection_info if k.startswith("P"))
    stats.detections_thing = sum(1 for k in detection_info if k.startswith("T"))
    stats.detections_vehicle = sum(1 for k in detection_info if k.startswith("V"))

    det_count = stats.edge_type_counts.get("DETECTED_IN", 0)
    if det_count > 0:
        for edge_type in ["INTERACTS_WITH", "CARRIES", "USES"]:
            stats.edge_type_ratios_vs_detected_in[edge_type] = stats.edge_type_counts.get(edge_type, 0) / det_count

    return detection_info


def validate_partition_and_uniqueness(
    dfs: Dict[str, pd.DataFrame],
    rels: pd.DataFrame,
    video_map: Dict[str, VideoInfo],
    detection_info: Dict[str, DetectionInfo],
    issues: IssueCollector,
    stats: DatasetStats,
) -> None:
    partition_by_loc: Dict[str, str] = {
        str(r.location_id): str(r.partition_id)
        for r in dfs["partitions.csv"].itertuples(index=False)
    }
    camera_partition: Dict[str, str] = {}

    for i, row in rels.iterrows():
        if str(row["type"]) != "LOCATED_AT":
            continue
        line = i + 2
        camera_id = str(row["source_id"])
        loc_id = str(row["destination_id"])
        loc_partition = partition_by_loc.get(loc_id)
        if not loc_partition:
            issues.error(f"LOCATED_AT references unknown location at line {line}: {loc_id}", "relation")
            continue
        prev = camera_partition.get(camera_id)
        if prev and prev != loc_partition:
            issues.error(f"Camera in multiple partitions via LOCATED_AT at line {line}: {camera_id}", "relation")
        camera_partition[camera_id] = loc_partition

    for vid, video in video_map.items():
        cam_partition = camera_partition.get(video.camera_id)
        if cam_partition and str(video.partition_id) != cam_partition:
            issues.error(
                f"partition(Location)->partition(Camera)->partition(Video) mismatch for video {vid}",
                "relation",
            )

    seen_entity_ids: Set[str] = set()
    seen_global_tw: Set[Tuple[str, str]] = set()
    entity_partition_declared: Dict[str, str] = {}

    for fname, (entity_id_col, global_id_col) in ENTITY_NODE_FILES.items():
        if fname not in dfs:
            continue
        stats.temporal_entity_counts[fname] = len(dfs[fname])

        for i, row in enumerate(dfs[fname].itertuples(index=False), start=1):
            line = i + 1
            entity_tw_id = str(getattr(row, entity_id_col))
            # id_global = str(getattr(row, global_id_col))
            entity_global_id = str(getattr(row, global_id_col))
            tw_id = str(row.tw_id)
            part = str(row.partition_id)

            if entity_tw_id in seen_entity_ids:
                issues.error(f"Duplicate Entity_TW ID in {fname} line {line}: {entity_tw_id}", "schema")
            seen_entity_ids.add(entity_tw_id)

            # pair = (id_global, tw_id)
            # if pair in seen_global_tw:
            #     issues.error(f"Duplicate (id_global, tw_id) in {fname} line {line}: {pair}", "schema")

            pair = (entity_global_id, tw_id)
            if pair in seen_global_tw:
                issues.error(f"Duplicate ({global_id_col}, tw_id) in {fname} line {line}: {pair}", "schema")
            seen_global_tw.add(pair)
            entity_partition_declared[entity_tw_id] = part

    stats.global_entity_counts["nodes_person.csv"] = len(dfs["nodes_person.csv"])
    stats.global_entity_counts["nodes_thing.csv"] = len(dfs["nodes_thing.csv"])
    stats.global_entity_counts["nodes_vehicle.csv"] = len(dfs["nodes_vehicle.csv"])

    for ent_id, det in detection_info.items():
        video = video_map.get(det.video_id)
        declared = entity_partition_declared.get(ent_id)
        if video and declared and declared != str(video.partition_id):
            issues.error(f"partition(Video)->partition(Entity_TW) mismatch for {ent_id}", "detection")

    expected_temporal_ids = set()
    for fname, (entity_id_col, _) in ENTITY_NODE_FILES.items():
        if fname in dfs:
            expected_temporal_ids.update(str(x) for x in dfs[fname][entity_id_col].tolist())

    missing_detection = sorted(expected_temporal_ids - set(detection_info.keys()))
    extra_detection = sorted(set(detection_info.keys()) - expected_temporal_ids)

    for ent_id in missing_detection[:20]:
        issues.error(f"Temporal node missing DETECTED_IN: {ent_id}", "detection")
    for ent_id in extra_detection[:20]:
        issues.error(f"DETECTED_IN references unknown temporal node: {ent_id}", "detection")

INDOOR_LOC_TYPES = {"Corridor", "Office", "Classroom", "Lobby"}

def validate_camera_indoor_consistency(
    dfs: Dict[str, pd.DataFrame],
    rels: pd.DataFrame,
    issues: IssueCollector,
) -> None:
    if "nodes_camera.csv" not in dfs or "nodes_location.csv" not in dfs:
        return

    location_type_map = build_location_type_map(dfs)
    camera_to_location = build_camera_location_map(rels)

    camera_rows = {
        str(r.camera_id): str(r.is_indoor).strip().lower()
        for r in dfs["nodes_camera.csv"].itertuples(index=False)
    }

    for cam_id, is_indoor_str in camera_rows.items():
        loc_id = camera_to_location.get(cam_id)
        if loc_id is None:
            continue

        loc_type = location_type_map.get(loc_id)
        if loc_type is None:
            issues.error(f"Camera {cam_id} maps to unknown location {loc_id}", "relation")
            continue

        expected = "true" if loc_type in INDOOR_LOC_TYPES else "false"
        if is_indoor_str != expected:
            issues.error(
                f"Camera indoor mismatch: camera_id={cam_id}, is_indoor={is_indoor_str}, "
                f"location_id={loc_id}, location_type={loc_type}, expected={expected}",
                "relation",
            )

def validate_vehicle_not_in_indoor_locations(
    dfs: Dict[str, pd.DataFrame],
    detection_info: Dict[str, DetectionInfo],
    issues: IssueCollector,
) -> None:
    location_type_map = build_location_type_map(dfs)
    for ent_id, det in detection_info.items():
        if not (ent_id.startswith("VH") or ent_id.startswith("V")):
            continue

        loc_type = location_type_map.get(det.location_id)
        if loc_type in INDOOR_LOC_TYPES:
            issues.error(
                f"Vehicle appears in indoor location: entity={ent_id}, "
                f"location_id={det.location_id}, location_type={loc_type}",
                "semantic",
            )

def validate_structural_edges(
    rels: pd.DataFrame,
    video_map: Dict[str, VideoInfo],
    issues: IssueCollector,
) -> None:
    node_partitions: Dict[str, str] = {}
    for v in video_map.values():
        node_partitions[v.video_id] = str(v.partition_id)

    for _, row in rels.iterrows():
        etype = str(row["type"])
        if etype in {"LOCATED_AT", "RECORDED_BY", "DETECTED_IN"}:
            node_partitions[str(row["source_id"])] = str(row["partition_id"])
            node_partitions[str(row["destination_id"])] = str(row["partition_id"])

    near_pairs: Set[Tuple[str, str, str]] = set()
    videos_by_camera: Dict[str, List[VideoInfo]] = {}
    for v in video_map.values():
        videos_by_camera.setdefault(v.camera_id, []).append(v)
    for cam in videos_by_camera:
        videos_by_camera[cam].sort(key=lambda x: x.start)

    for i, row in rels.iterrows():
        line = i + 2
        etype = str(row["type"])
        src = str(row["source_id"])
        dst = str(row["destination_id"])
        part = str(row["partition_id"])

        if etype == "NEAR_BY":
            if src not in node_partitions or dst not in node_partitions:
                issues.error(f"NEAR_BY endpoint missing at line {line}: {src}->{dst}", "relation")
                continue
            src_part = node_partitions.get(src)
            dst_part = node_partitions.get(dst)
            if src_part != dst_part or src_part != part:
                issues.error(
                    f"NEAR_BY partition mismatch at line {line}: {src}({src_part})->{dst}({dst_part}), rel={part}",
                    "relation",
                )
            near_pairs.add((src, dst, part))
            continue

        if etype == "NEXT_TO":
            v1 = video_map.get(src)
            v2 = video_map.get(dst)
            if not v1 or not v2:
                issues.error(f"NEXT_TO missing video endpoint at line {line}: {src}->{dst}", "relation")
                continue
            if v1.camera_id != v2.camera_id:
                issues.error(f"NEXT_TO camera mismatch at line {line}: {src}->{dst}", "relation")
                continue

            if str(v1.partition_id) != str(v2.partition_id):
                issues.error(
                    f"NEXT_TO videos cross partitions unexpectedly at line {line}: "
                    f"{src}({v1.partition_id}) -> {dst}({v2.partition_id})",
                    "relation",
                )

            if str(part) != str(v1.partition_id):
                issues.error(
                    f"NEXT_TO partition_id mismatch at line {line}: rel.partition_id={part}, "
                    f"video.partition_id={v1.partition_id} for {src}->{dst}",
                    "relation",
                )

            ordered = videos_by_camera.get(v1.camera_id, [])
            is_consecutive = False
            for idx in range(len(ordered) - 1):
                if ordered[idx].video_id == src and ordered[idx + 1].video_id == dst:
                    is_consecutive = True
                    break
            if not is_consecutive:
                issues.error(f"NEXT_TO must connect consecutive videos at line {line}: {src}->{dst}", "relation")

    for src, dst, part in near_pairs:
        if (dst, src, part) not in near_pairs:
            issues.error(f"NEAR_BY reverse missing for {src}->{dst} in partition {part}", "relation")


def validate_semantic_edges(
    rels: pd.DataFrame,
    detection_info: Dict[str, DetectionInfo],
    issues: IssueCollector,
    stats: DatasetStats,
) -> None:
    interactions_seen = set()

    for i, row in rels.iterrows():
        etype = str(row["type"])
        if etype not in {"CARRIES", "USES", "INTERACTS_WITH"}:
            continue

        line = i + 2
        src = str(row["source_id"])
        dst = str(row["destination_id"])

        src_det = detection_info.get(src)
        dst_det = detection_info.get(dst)

        if not src_det or not dst_det:
            issues.error(f"Missing detection for semantic edge at line {line}: {src}->{dst} ({etype})", "relation")
            continue

        if not (
            src_det.date == dst_det.date and
            src_det.tw_id == dst_det.tw_id and
            src_det.video_id == dst_det.video_id and
            src_det.camera_id == dst_det.camera_id and
            src_det.location_id == dst_det.location_id
        ):
            issues.error(f"Semantic edge not co-contextual at line {line}: {src}->{dst} ({etype})", "semantic")
        else:
            stats.semantic_edge_same_context_ok += 1

        row_date = str(row["date"])
        row_tw_id = str(row["tw_id"])
        row_camera_id = str(row["camera_id"])
        row_location_id = str(row["location_id"])

        if row_date != src_det.date:
            issues.error(
                f"Semantic edge date mismatch at line {line}: row.date={row_date}, detection.date={src_det.date}",
                "semantic",
            )
        if row_tw_id != src_det.tw_id:
            issues.error(
                f"Semantic edge tw_id mismatch at line {line}: row.tw_id={row_tw_id}, detection.tw_id={src_det.tw_id}",
                "semantic",
            )
        if row_camera_id != src_det.camera_id:
            issues.error(
                f"Semantic edge camera_id mismatch at line {line}: row.camera_id={row_camera_id}, "
                f"detection.camera_id={src_det.camera_id}",
                "semantic",
            )
        if row_location_id != src_det.location_id:
            issues.error(
                f"Semantic edge location_id mismatch at line {line}: row.location_id={row_location_id}, "
                f"detection.location_id={src_det.location_id}",
                "semantic",
            )

        overlap_start = max(src_det.start_time, dst_det.start_time)
        overlap_end = min(src_det.end_time, dst_det.end_time)
        if not overlap_start < overlap_end:
            issues.error(f"Semantic edge has no positive overlap at line {line}: {src}->{dst} ({etype})", "relation")
            continue

        row_start = parse_ts(row["start_time"])
        row_end = parse_ts(row["end_time"])
        if not (overlap_start <= row_start < row_end <= overlap_end):
            issues.error(
                f"Semantic edge interval not inside detection overlap at line {line}: {src}->{dst} ({etype})",
                "relation",
            )
        else:
            stats.semantic_edge_overlap_ok += 1

        if etype == "INTERACTS_WITH":
            interactions_seen.add((src, dst, row["start_time"], row["end_time"]))

    for src, dst, start_time, end_time in interactions_seen:
        if (dst, src, start_time, end_time) not in interactions_seen:
            issues.error(f"INTERACTS_WITH reverse missing: {src}->{dst} [{start_time}, {end_time}]", "relation")


def compute_co_context_stats(rels: pd.DataFrame, stats: DatasetStats) -> None:
    det = rels[rels["type"] == "DETECTED_IN"].copy()

    persons = det[det["source_id"].str.startswith("P")].copy()
    things = det[det["source_id"].str.startswith("T")].copy()
    vehicles = det[det["source_id"].str.startswith("V")].copy()

    pair_keys = ["date", "tw_id", "location_id", "camera_id", "destination_id"]

    person_vehicle = persons.merge(vehicles, on=pair_keys, suffixes=("_p", "_v"))
    person_thing = persons.merge(things, on=pair_keys, suffixes=("_p", "_t"))
    person_person = persons.merge(persons, on=pair_keys, suffixes=("_a", "_b"))
    person_person = person_person[person_person["source_id_a"] < person_person["source_id_b"]]

    stats.person_vehicle_same_context_pairs = len(person_vehicle)
    stats.person_thing_same_context_pairs = len(person_thing)
    stats.person_person_same_context_pairs = len(person_person)


def validate_partition_self_containment(root: Path, issues: IssueCollector, stats: DatasetStats) -> None:
    by_partition_root = root / "by_partition"
    if not by_partition_root.exists():
        return

    stats.by_partition_dirs = 0
    stats.by_partition_rel_rows_total = 0

    for pdir in sorted(x for x in by_partition_root.iterdir() if x.is_dir()):
        stats.by_partition_dirs += 1
        local_ids: Set[str] = set()

        node_files = [
            "nodes_location.csv",
            "nodes_camera.csv",
            "nodes_video.csv",
            "nodes_person_TW.csv",
            "nodes_thing_TW.csv",
            "nodes_vehicle_TW.csv",
        ]
        for nf in node_files:
            path = pdir / nf
            if not path.exists():
                continue
            ndf = read_csv_safe(path)
            if ndf.empty:
                continue
            id_col = ndf.columns[0]
            local_ids.update(str(x) for x in ndf[id_col].tolist())

        rel_path = pdir / "rels.csv"
        if not rel_path.exists():
            continue
        rdf = read_csv_safe(rel_path)
        stats.by_partition_rel_rows_total += len(rdf)

        for i, row in rdf.iterrows():
            src = str(row["source_id"])
            dst = str(row["destination_id"])
            if src not in local_ids or dst not in local_ids:
                missing = []
                if src not in local_ids:
                    missing.append(f"source_id={src}")
                if dst not in local_ids:
                    missing.append(f"destination_id={dst}")
                issues.error(
                    f"Orphan edge in {pdir.name}/rels.csv line {i + 2}: "
                    f"type={row['type']}, partition_id={row['partition_id']}, "
                    f"{', '.join(missing)}",
                    "relation",
                )


# =============================================================================
# STORAGE-LAYER VALIDATION
# =============================================================================

def validate_storage_layers(root: Path, dfs: Dict[str, pd.DataFrame], issues: IssueCollector, stats: DatasetStats) -> None:
    # [1] Root dataset = global view / baseline
    required_root = set(REQUIRED_ROOT_FILES)
    present_root = set(stats.root_files_present)
    if present_root != required_root:
        missing = sorted(required_root - present_root)
        extra = sorted(present_root - required_root)
        if missing:
            issues.error(f"Root dataset incomplete, missing files: {missing}", "storage")
        if extra:
            issues.warn(f"Root dataset has extra files: {extra}")

    # [2] by_partition = execution view for LocalEval -> Stitching
    by_partition_root = root / "by_partition"
    expected_partition_ids = set(str(x) for x in dfs["partitions.csv"]["partition_id"].astype(str).unique().tolist())
    stats.by_partition_expected_dirs = len(expected_partition_ids)

    if by_partition_root.exists():
        actual_partition_dir_names = set()
        rel_rows_total = 0

        for pdir in sorted(x for x in by_partition_root.iterdir() if x.is_dir()):
            actual_partition_dir_names.add(pdir.name)

        for filename in STORAGE_PARTITION_AGG_FILES:
            root_path = root / filename
            if not root_path.exists():
                issues.error(f"Root file missing for by_partition comparison: {filename}", "storage")
                continue

            part_files = collect_partition_files(root, filename)
            if not part_files:
                issues.error(f"No partition-local files found for: {filename}", "storage")
                continue

            root_df = read_csv_safe(root_path)
            part_df = concat_csvs(part_files)

            if len(root_df) != len(part_df):
                issues.error(
                    f"Row-count mismatch for {filename}: root={len(root_df)} vs by_partition_sum={len(part_df)}",
                    "storage",
                )
                continue

            if df_signature(root_df) != df_signature(part_df):
                issues.error(
                    f"Content mismatch for {filename}: root CSV is not exactly the union of by_partition files",
                    "storage",
                )

            if filename == "rels.csv":
                rel_rows_total = len(part_df)

        root_rel_rows = len(dfs["rels.csv"])
        stats.by_partition_rel_rows_total = rel_rows_total
        stats.by_partition_rel_rows_match_root = rel_rows_total == root_rel_rows
        if rel_rows_total != root_rel_rows:
            issues.error(
                f"by_partition rel rows ({rel_rows_total}) != root rel rows ({root_rel_rows})",
                "storage",
            )
        if stats.by_partition_dirs != len(actual_partition_dir_names):
            stats.by_partition_dirs = len(actual_partition_dir_names)
        if len(actual_partition_dir_names) != len(expected_partition_ids):
            issues.warn(
                f"by_partition dir count ({len(actual_partition_dir_names)}) != unique partition ids ({len(expected_partition_ids)})"
            )
    else:
        issues.warn("Missing by_partition/ directory; cannot validate execution-view storage layer")

    # [3] date=YYYY-MM-DD = day-level temporal anchor (auxiliary layer)
    date_dirs = sorted([p for p in root.glob("date=*") if p.is_dir()])
    expected_dates = set(str(x) for x in dfs["nodes_timewindow.csv"]["date"].astype(str).unique().tolist())
    stats.date_anchor_dates_expected = sorted(expected_dates)
    stats.date_anchor_expected_dirs = len(expected_dates)
    stats.date_anchor_dirs = len(date_dirs)
    stats.date_anchor_dates_found = sorted(d.name.split("=", 1)[1] for d in date_dirs if "=" in d.name)

    if date_dirs:
        found_dates = set(stats.date_anchor_dates_found)
        missing_dates = sorted(expected_dates - found_dates)
        extra_dates = sorted(found_dates - expected_dates)

        if missing_dates:
            issues.error(f"Missing date anchor directories for dates: {missing_dates}", "storage")
        if extra_dates:
            issues.warn(f"Extra date anchor directories found: {extra_dates}")

        total_loc_dirs = 0

        for ddir in date_dirs:
            anchor_date = ddir.name.split("=", 1)[1]
            loc_dirs = sorted(x for x in ddir.iterdir() if x.is_dir() and x.name.startswith("loc="))
            total_loc_dirs += len(loc_dirs)

            for ldir in loc_dirs:
                loc_id = ldir.name.split("=", 1)[1]

                for csv_path in ldir.glob("*.csv"):
                    try:
                        cdf = read_csv_safe(csv_path)
                    except Exception as e:
                        issues.error(f"Failed reading {csv_path}: {e}", "storage")
                        continue

                    if cdf.empty:
                        continue

                    if "date" in cdf.columns:
                        mism = cdf[cdf["date"].astype(str) != anchor_date]
                        if not mism.empty:
                            issues.error(
                                f"Day-anchor violation in {csv_path}: found rows outside date={anchor_date}",
                                "storage",
                            )

                    if csv_path.name == "rels.csv" and "location_id" in cdf.columns:
                        bad_loc = cdf[cdf["location_id"].astype(str) != loc_id]
                        if not bad_loc.empty:
                            issues.error(
                                f"Location anchor violation in {csv_path}: rows with location_id != {loc_id}",
                                "storage",
                            )

        stats.date_anchor_loc_dirs = total_loc_dirs
    else:
        issues.warn("No date=YYYY-MM-DD directories found; cannot validate day-level anchor layer")

# =============================================================================
# SUMMARY / REPORTING
# =============================================================================

def build_summary(issues: IssueCollector) -> ValidationSummary:
    s = ValidationSummary()

    if "relation" in issues.categories:
        s.relation_valid = "FAIL"
    if "detection" in issues.categories:
        s.detection_valid = "FAIL"
    if "semantic" in issues.categories:
        s.semantic_valid = "FAIL"
    if "schema" in issues.categories:
        s.schema_valid = "FAIL"
    if "file" in issues.categories:
        s.file_valid = "FAIL"
    if "storage" in issues.categories:
        s.storage_valid = "FAIL"

    s.total_errors = len(issues.errors)
    s.total_warnings = len(issues.warnings)
    return s


def print_stats(stats: DatasetStats) -> None:
    print_section("DATASET FILE COUNTS")
    for k, v in stats.file_row_counts.items():
        print(f"{k:24s}: {v}")

    print_section("ROOT FILE CHECK")
    print(f"present: {len(stats.root_files_present)}")
    print(f"missing: {len(stats.root_files_missing)}")
    if stats.root_files_missing:
        for x in stats.root_files_missing:
            print(f"- {x}")

    print_section("NODE COUNTS")
    for k, v in stats.global_entity_counts.items():
        print(f"{k:24s}: {v}")
    for k, v in stats.temporal_entity_counts.items():
        print(f"{k:24s}: {v}")

    print_section("EDGE TYPE DISTRIBUTION")
    for k, v in sorted(stats.edge_type_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k:16s}: {v}")

    print_section("EDGE RATIOS VS DETECTED_IN")
    det = stats.edge_type_counts.get("DETECTED_IN", 0)
    print(f"DETECTED_IN total : {det}")
    for edge_type in ["INTERACTS_WITH", "CARRIES", "USES"]:
        ratio = stats.edge_type_ratios_vs_detected_in.get(edge_type, 0.0)
        print(f"{edge_type:16s}: {_fmt_ratio(ratio)}")

    print_section("DETECTION BREAKDOWN")
    print(f"total detections   : {stats.detections_total}")
    print(f"person detections  : {stats.detections_person}")
    print(f"thing detections   : {stats.detections_thing}")
    print(f"vehicle detections : {stats.detections_vehicle}")

    print_section("SAME-CONTEXT CANDIDATE COUNTS")
    print(f"person-person pairs  : {stats.person_person_same_context_pairs}")
    print(f"person-thing pairs   : {stats.person_thing_same_context_pairs}")
    print(f"person-vehicle pairs : {stats.person_vehicle_same_context_pairs}")

    print_section("SEMANTIC EDGE CHECK COUNTS")
    print(f"same-context ok : {stats.semantic_edge_same_context_ok}")
    print(f"overlap ok      : {stats.semantic_edge_overlap_ok}")

    print_section("STORAGE LAYER CHECKS")
    print("[1] ROOT DATASET = global view / baseline")
    print(f"root files present           : {len(stats.root_files_present)}")
    print(f"root files missing           : {len(stats.root_files_missing)}")

    print("\n[2] by_partition/ = execution view for LocalEval -> Stitching")
    print(f"partition dirs found         : {stats.by_partition_dirs}")
    print(f"partition dirs expected      : {stats.by_partition_expected_dirs}")
    print(f"partition rel rows total     : {stats.by_partition_rel_rows_total}")
    print(f"partition rel rows == root   : {stats.by_partition_rel_rows_match_root}")

    print("\n[3] date=YYYY-MM-DD = day-level temporal anchor")
    print(f"date dirs found              : {stats.date_anchor_dirs}")
    print(f"date dirs expected           : {stats.date_anchor_expected_dirs}")
    print(f"date values expected         : {stats.date_anchor_dates_expected}")
    print(f"date values found            : {stats.date_anchor_dates_found}")
    print(f"date-layer loc dirs          : {stats.date_anchor_loc_dirs}")
    print("date-layer mode              : anchor-placement only")


# =============================================================================
# MAIN
# =============================================================================

def verify_dataset(data_dir: str) -> ValidationSummary:
    root = Path(data_dir)
    issues = IssueCollector()
    stats = DatasetStats()

    dfs = load_root_csvs(root, issues, stats)
    validate_required_columns(dfs, issues)

    missing_roots = [f for f in REQUIRED_ROOT_FILES if f not in dfs]
    if missing_roots:
        summary = build_summary(issues)
        print("Validation Summary:")
        print(summary)
        print("\nCannot continue because required files are missing.")
        return summary

    # tw_map = validate_timewindows(dfs["nodes_timewindow.csv"], issues)
    # video_map = validate_videos(dfs["nodes_video.csv"], issues)

    # detection_info = validate_edges(dfs["rels.csv"], video_map, tw_map, issues, stats)
    # validate_partition_and_uniqueness(dfs, dfs["rels.csv"], video_map, detection_info, issues, stats)
    # validate_structural_edges(dfs["rels.csv"], video_map, issues)
    # validate_semantic_edges(dfs["rels.csv"], detection_info, issues, stats)
    # compute_co_context_stats(dfs["rels.csv"], stats)
    # validate_partition_self_containment(root, issues, stats)
    # validate_storage_layers(root, dfs, issues, stats)

    tw_map = validate_timewindows(dfs["nodes_timewindow.csv"], issues)
    video_map = validate_videos(dfs["nodes_video.csv"], issues)

    camera_to_location = build_camera_location_map(dfs["rels.csv"])

    detection_info = validate_edges(
        dfs["rels.csv"],
        video_map,
        tw_map,
        camera_to_location,
        issues,
        stats,
    )

    validate_partition_and_uniqueness(dfs, dfs["rels.csv"], video_map, detection_info, issues, stats)
    validate_camera_indoor_consistency(dfs, dfs["rels.csv"], issues)
    validate_vehicle_not_in_indoor_locations(dfs, detection_info, issues)
    validate_structural_edges(dfs["rels.csv"], video_map, issues)
    validate_semantic_edges(dfs["rels.csv"], detection_info, issues, stats)
    compute_co_context_stats(dfs["rels.csv"], stats)
    validate_partition_self_containment(root, issues, stats)
    validate_storage_layers(root, dfs, issues, stats)

    summary = build_summary(issues)

    print("Validation Summary:")
    print(summary)
    print_stats(stats)

    if issues.errors:
        print_section("ERRORS (first 100)")
        for e in issues.errors[:100]:
            print("-", e)

    if issues.warnings:
        print_section("WARNINGS (first 100)")
        for w in issues.warnings[:100]:
            print("-", w)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final validator for generated surveillance KG dataset: data correctness + storage-layer consistency"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--report_json", default="", help="Optional path to save summary/errors/warnings/stats as JSON")
    args = parser.parse_args()

    summary = verify_dataset(args.data_dir)

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": asdict(summary),
            "status": "PASS" if summary.total_errors == 0 else "FAIL",
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report saved to: {out}")
