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

    validate_edges(
        dfs["rels.csv"],
        video_map,
        tw_map,
        issues
    )

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