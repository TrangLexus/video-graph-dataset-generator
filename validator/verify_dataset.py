#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# =========================
# CONSTANTS
# =========================

TEMPORAL_EDGE_TYPES = {"DETECTED_IN", "CARRIES", "USES", "INTERACTS_WITH"}
STRUCTURAL_EDGE_TYPES = {"LOCATED_AT", "RECORDED_BY", "NEAR_BY", "NEXT_TO"}
ALL_EDGE_TYPES = TEMPORAL_EDGE_TYPES | STRUCTURAL_EDGE_TYPES


# =========================
# DATA MODELS
# =========================

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
    ts_start: datetime
    ts_end: datetime
    video_id: str
    camera_id: str
    location_id: str
    date: str
    tw_id: str
    partition_id: str


# =========================
# ISSUE COLLECTOR (NEW)
# =========================

@dataclass
class IssueCollector:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    categories: Set[str] = field(default_factory=set)

    def error(self, msg: str, category: str = "general"):
        self.errors.append(msg)
        self.categories.add(category)

    def warn(self, msg: str):
        self.warnings.append(msg)


# =========================
# SUMMARY
# =========================

@dataclass
class ValidationSummary:
    relation_valid: str = "PASS"
    detection_valid: str = "PASS"
    semantic_valid: str = "PASS"
    schema_valid: str = "PASS"

    total_errors: int = 0
    total_warnings: int = 0


# =========================
# UTIL
# =========================

def parse_ts(x: str) -> datetime:
    return datetime.fromisoformat(x)


def parse_time(date: str, t: str) -> datetime:
    return datetime.fromisoformat(f"{date} {t}")


# =========================
# CORE VALIDATION
# =========================

def validate_timewindows(df: pd.DataFrame, issues: IssueCollector):
    tw_map: Dict[Tuple[str, str], TimeWindowInfo] = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        line = int(i) + 1

        try:
            date = str(row.date)
            tw_id = str(row.tw_id)

            start = parse_time(date, str(row.start_time))
            end = parse_time(date, str(row.end_time))

            if start >= end:
                issues.error(f"TW invalid interval line {line}", "schema")

            tw_map[(date, tw_id)] = TimeWindowInfo(
                date=date, tw_id=tw_id, start=start, end=end
            )

        except Exception as e:
            issues.error(f"TW parse error line {line}: {e}", "schema")

    return tw_map


def validate_videos(df: pd.DataFrame, issues: IssueCollector):
    videos: Dict[str, VideoInfo] = {}

    for i, row in enumerate(df.itertuples(index=False), start=1):
        line = int(i) + 1

        try:
            vid = str(row.video_id)
            date = str(row.date)

            start = parse_time(date, str(row.start_time))
            end = parse_time(date, str(row.end_time))

            if start >= end:
                issues.error(f"Video invalid interval line {line}", "schema")

            videos[vid] = VideoInfo(
                video_id=vid,
                camera_id=str(row.camera_id),
                partition_id=str(row.partition_id),
                date=date,
                start=start,
                end=end,
            )

        except Exception as e:
            issues.error(f"Video parse error {line}: {e}", "schema")

    return videos


def validate_edges(rels: pd.DataFrame,
                   video_map: Dict[str, VideoInfo],
                   tw_map: Dict[Tuple[str, str], TimeWindowInfo],
                   issues: IssueCollector):

    detection_info: Dict[str, DetectionInfo] = {}

    for i, row in rels.iterrows():
        line = i + 2
        etype = row["type"]

        src = str(row["source_id"])
        dst = str(row["destination_id"])

        ts_start = row["ts_start"]
        ts_end = row["ts_end"]

        dt_start = parse_ts(ts_start) if ts_start else None
        dt_end = parse_ts(ts_end) if ts_end else None

        # =========================
        # NEXT_TO (FIXED LOGIC)
        # =========================
        if etype == "NEXT_TO":
            if dt_start and dt_end and dt_start != dt_end:
                issues.error(
                    f"NEXT_TO must be point event line {line}",
                    "relation"
                )
            continue

        # =========================
        # DETECTED_IN
        # =========================
        if etype == "DETECTED_IN":
            if not (dt_start and dt_end):
                issues.error("DETECTED_IN missing time", "detection")
                continue

            video = video_map.get(dst)
            if video:
                if not (video.start <= dt_start < dt_end <= video.end):
                    issues.error("DETECTED_IN outside video", "detection")

            detection_info[src] = DetectionInfo(
                entity_tw_id=src,
                ts_start=dt_start,
                ts_end=dt_end,
                video_id=dst,
                camera_id="",
                location_id="",
                date=str(row["date"]),
                tw_id=str(row["tw_id"]),
                partition_id=str(row["partition_id"]),
            )

            continue

        # =========================
        # SEMANTIC EDGES
        # =========================
        if etype in {"CARRIES", "USES", "INTERACTS_WITH"}:
            src_det = detection_info.get(src)
            dst_det = detection_info.get(dst)

            if not src_det or not dst_det:
                issues.error("Missing detection for semantic edge", "relation")
                continue

            overlap_start = max(src_det.ts_start, dst_det.ts_start)
            overlap_end = min(src_det.ts_end, dst_det.ts_end)

            if not overlap_start < overlap_end:
                issues.error("No overlap in semantic edge", "relation")

    return detection_info


def validate_partition_and_uniqueness(
    dfs: Dict[str, pd.DataFrame],
    rels: pd.DataFrame,
    video_map: Dict[str, VideoInfo],
    detection_info: Dict[str, DetectionInfo],
    issues: IssueCollector,
):
    partition_by_loc: Dict[str, str] = {
        str(r.loc_id): str(r.partition_id)
        for r in dfs["partitions.csv"].itertuples(index=False)
    }
    camera_partition: Dict[str, str] = {}

    for i, row in rels.iterrows():
        if row["type"] != "LOCATED_AT":
            continue
        line = i + 2
        camera_id = str(row["source_id"])
        loc_id = str(row["destination_id"])
        loc_partition = partition_by_loc.get(loc_id)
        if not loc_partition:
            issues.error(
                f"LOCATED_AT references unknown location at line {line}: {loc_id}",
                "relation",
            )
            continue
        prev = camera_partition.get(camera_id)
        if prev and prev != loc_partition:
            issues.error(
                f"Camera in multiple partitions via LOCATED_AT at line {line}: {camera_id}",
                "relation",
            )
        camera_partition[camera_id] = loc_partition

    for vid, video in video_map.items():
        cam_partition = camera_partition.get(video.camera_id)
        if cam_partition and str(video.partition_id) != cam_partition:
            issues.error(
                f"partition(Location)->partition(Camera)->partition(Video) mismatch for video {vid}",
                "relation",
            )

    entity_files = ["nodes_person_TW.csv", "nodes_thing_TW.csv", "nodes_vehicle_TW.csv"]
    seen_entity_ids: Set[str] = set()
    seen_global_tw: Set[Tuple[str, str]] = set()
    entity_partition_declared: Dict[str, str] = {}

    for fname in entity_files:
        if fname not in dfs:
            continue
        for i, row in enumerate(dfs[fname].itertuples(index=False), start=1):
            line = i + 1
            entity_tw_id = str(row.entity_tw_id)
            id_global = str(row.id_global)
            tw_id = str(row.tw_id)
            part = str(row.partition_id)

            if entity_tw_id in seen_entity_ids:
                issues.error(f"Duplicate Entity_TW ID in {fname} line {line}: {entity_tw_id}", "schema")
            seen_entity_ids.add(entity_tw_id)

            pair = (id_global, tw_id)
            if pair in seen_global_tw:
                issues.error(f"Duplicate (id_global, tw_id) in {fname} line {line}: {pair}", "schema")
            seen_global_tw.add(pair)
            entity_partition_declared[entity_tw_id] = part

    for ent_id, det in detection_info.items():
        video = video_map.get(det.video_id)
        declared = entity_partition_declared.get(ent_id)
        if video and declared and declared != str(video.partition_id):
            issues.error(
                f"partition(Video)->partition(Entity_TW) mismatch for {ent_id}",
                "detection",
            )


def validate_structural_edges(
    rels: pd.DataFrame,
    video_map: Dict[str, VideoInfo],
    issues: IssueCollector,
):
    node_partitions: Dict[str, str] = {}
    for v in video_map.values():
        node_partitions[v.video_id] = str(v.partition_id)

    for i, row in rels.iterrows():
        etype = row["type"]
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
        etype = row["type"]
        src = str(row["source_id"])
        dst = str(row["destination_id"])
        part = str(row["partition_id"])

        if etype == "NEAR_BY":
            if src not in node_partitions or dst not in node_partitions:
                issues.error(
                    f"NEAR_BY endpoint missing at line {line}: {src}->{dst}",
                    "relation",
                )
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


def validate_partition_self_containment(root: Path, issues: IssueCollector):
    by_partition_root = root / "by_partition"
    if not by_partition_root.exists():
        return

    for pdir in sorted(x for x in by_partition_root.iterdir() if x.is_dir()):
        # Collect all node IDs physically present in this partition directory.
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
            ndf = pd.read_csv(path, dtype=str, keep_default_na=False)
            if ndf.empty:
                continue
            id_col = ndf.columns[0]
            local_ids.update(str(x) for x in ndf[id_col].tolist())

        rel_path = pdir / "rels.csv"
        if not rel_path.exists():
            continue
        rdf = pd.read_csv(rel_path, dtype=str, keep_default_na=False)
        for i, row in rdf.iterrows():
            # Every relation in by_partition must be self-contained: both endpoints
            # must resolve to IDs that exist in the same partition directory.
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


# =========================
# SUMMARY
# =========================

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

    s.total_errors = len(issues.errors)
    s.total_warnings = len(issues.warnings)

    if s.total_errors > 0 and all(
        x == "PASS" for x in [s.relation_valid, s.detection_valid, s.semantic_valid, s.schema_valid]
    ):
        s.schema_valid = "FAIL"

    return s


# =========================
# MAIN
# =========================

def verify_dataset(data_dir: str):
    root = Path(data_dir)
    issues = IssueCollector()

    dfs = {
        f.name: pd.read_csv(f, dtype=str, keep_default_na=False)
        for f in root.glob("*.csv")
    }

    tw_map = validate_timewindows(dfs["nodes_timewindow.csv"], issues)
    video_map = validate_videos(dfs["nodes_video.csv"], issues)

    detection_info = validate_edges(
        dfs["rels.csv"],
        video_map,
        tw_map,
        issues
    )

    validate_partition_and_uniqueness(
        dfs,
        dfs["rels.csv"],
        video_map,
        detection_info,
        issues,
    )
    validate_structural_edges(
        dfs["rels.csv"],
        video_map,
        issues,
    )
    validate_partition_self_containment(root, issues)

    summary = build_summary(issues)

    print("Validation Summary:")
    print(summary)

    if issues.errors:
        print("\nErrors:")
        for e in issues.errors[:20]:
            print("-", e)

    return summary


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)

    args = parser.parse_args()

    verify_dataset(args.data_dir)
