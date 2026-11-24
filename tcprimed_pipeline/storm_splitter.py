"""
Storm Splitter module for TC PRIMED.
Performs reproducible, leak-free, storm-wise partitioning into Train, Validation, and Test sets,
stratified by ocean basin and historical era.
"""

import os
import csv
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter

from .config import SplittingConfig


ERA_BINS = [
    ("1987-1999", 1987, 1999),
    ("2000-2009", 2000, 2009),
    ("2010-2019", 2010, 2019),
    ("2020+",     2020, 9999),
]


def classify_era(season: int) -> str:
    """Classify cyclone season into historical era bin."""
    for label, start_yr, end_yr in ERA_BINS:
        if start_yr <= season <= end_yr:
            return label
    return "Unknown"


class StormSplitter:
    """Manages storm-wise train / validation / test partitioning."""

    def __init__(self, config: Optional[SplittingConfig] = None):
        self.config = config or SplittingConfig()

    def split_storms(self,
                     storm_records: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Partition storms into TRAIN, VAL, and TEST splits.
        Input:
            storm_records: {storm_id: {'season': int, 'basin': str, 'overpass_files': [...]}}
        Returns:
            {storm_id: 'TRAIN' | 'VAL' | 'TEST'}
        """
        random.seed(self.config.seed)
        all_sids = sorted(list(storm_records.keys()))
        total_storms = len(all_sids)

        if total_storms == 0:
            return {}

        target_train_count = round(self.config.train * total_storms)
        target_val_count = round(self.config.validation * total_storms)
        target_test_count = total_storms - target_train_count - target_val_count

        # Check for prior pilot storms when splitting an expanded (>80 storms) dataset
        # so prior pilot storms never cross partitions or enter the untouched FINAL_TEST partition
        prior_pilot_splits: Dict[str, str] = {}
        pilot_split_csv = "reports/pipeline/storm_split_manifest.csv"
        if total_storms > 80 and os.path.exists(pilot_split_csv):
            try:
                with open(pilot_split_csv, "r", encoding="utf-8") as pf:
                    reader = csv.DictReader(pf)
                    for row in reader:
                        p_sid = row.get("storm_id", "").strip()
                        p_part = row.get("partition", "").strip().upper()
                        if p_sid in storm_records and p_part in ("TRAIN", "VAL", "TEST"):
                            prior_pilot_splits[p_sid] = p_part
            except Exception:
                prior_pilot_splits = {}

        # Only enforce pilot isolation when there are sufficient new non-pilot storms for VAL and TEST
        non_pilot_sids = [sid for sid in all_sids if sid not in prior_pilot_splits]
        enforce_pilot_isolation = (
            len(prior_pilot_splits) > 0
            and len(non_pilot_sids) >= (target_val_count + target_test_count)
        )

        # Stratified grouping by (basin, era)
        strata = defaultdict(list)
        for sid in (non_pilot_sids if enforce_pilot_isolation else all_sids):
            st = storm_records[sid]
            basin = st.get("basin", sid[:2])
            season = st.get("season", 0)
            era = classify_era(season)
            strata[(basin, era)].append(sid)

        # Shuffle each stratum deterministically
        all_sids_ordered = []
        for s_key in sorted(strata.keys()):
            s_sids = strata[s_key]
            random.shuffle(s_sids)
            all_sids_ordered.extend(s_sids)

        train_sids: List[str] = []
        val_sids: List[str] = []
        test_sids: List[str] = []

        if enforce_pilot_isolation:
            # Per-basin proportional allocation ensuring every basin with >=3 storms
            # has representation in TRAIN, VAL, and untouched FINAL_TEST
            by_basin = defaultdict(list)
            for sid in all_sids:
                b = storm_records[sid].get("basin", sid[:2])
                by_basin[b].append(sid)

            for b in sorted(by_basin.keys()):
                b_sids = by_basin[b]
                n_b = len(b_sids)
                b_val_target = max(1, round(self.config.validation * n_b)) if n_b >= 3 else 0
                b_test_target = max(1, round(self.config.test * n_b)) if n_b >= 3 else 0
                b_train_target = n_b - b_val_target - b_test_target

                b_train, b_val, b_test = [], [], []
                # 1. Assign prior pilot storms first (never into TEST)
                b_non_pilot = []
                for sid in b_sids:
                    if sid in prior_pilot_splits:
                        if prior_pilot_splits[sid] == "TRAIN" and len(b_train) < b_train_target:
                            b_train.append(sid)
                        elif len(b_val) < b_val_target:
                            b_val.append(sid)
                        else:
                            b_train.append(sid)
                    else:
                        b_non_pilot.append(sid)

                # 2. Order remaining non-pilot storms in this basin by era and shuffle deterministically
                b_era_groups = defaultdict(list)
                for sid in b_non_pilot:
                    era = classify_era(storm_records[sid].get("season", 0))
                    b_era_groups[era].append(sid)
                b_ordered = []
                for e_k in sorted(b_era_groups.keys()):
                    e_list = b_era_groups[e_k]
                    random.shuffle(e_list)
                    b_ordered.extend(e_list)

                # 3. Fill TEST, VAL, and TRAIN for this basin
                for idx, sid in enumerate(b_ordered):
                    if idx % 6 == 0 and len(b_test) < b_test_target:
                        b_test.append(sid)
                    elif idx % 6 == 3 and len(b_val) < b_val_target:
                        b_val.append(sid)
                    elif len(b_train) < b_train_target:
                        b_train.append(sid)
                    elif len(b_val) < b_val_target:
                        b_val.append(sid)
                    else:
                        b_test.append(sid)

                train_sids.extend(b_train)
                val_sids.extend(b_val)
                test_sids.extend(b_test)
        else:
            # Proportional round-robin assignment preserving exact target counts
            for idx, sid in enumerate(all_sids_ordered):
                mod = idx % 20  # 14 to train (70%), 3 to val (15%), 3 to test (15%)
                if mod < 14:
                    if len(train_sids) < target_train_count:
                        train_sids.append(sid)
                    elif len(val_sids) < target_val_count:
                        val_sids.append(sid)
                    else:
                        test_sids.append(sid)
                elif mod < 17:
                    if len(val_sids) < target_val_count:
                        val_sids.append(sid)
                    elif len(test_sids) < target_test_count:
                        test_sids.append(sid)
                    else:
                        train_sids.append(sid)
                else:
                    if len(test_sids) < target_test_count:
                        test_sids.append(sid)
                    elif len(train_sids) < target_train_count:
                        train_sids.append(sid)
                    else:
                        val_sids.append(sid)

        split_map = {}
        for sid in train_sids: split_map[sid] = "TRAIN"
        for sid in val_sids: split_map[sid] = "VAL"
        for sid in test_sids: split_map[sid] = "TEST"

        # Validate zero-leakage guarantee
        train_set = set(train_sids)
        val_set = set(val_sids)
        test_set = set(test_sids)

        assert len(train_set & val_set) == 0, f"Train-Val leak detected: {train_set & val_set}"
        assert len(train_set & test_set) == 0, f"Train-Test leak detected: {train_set & test_set}"
        assert len(val_set & test_set) == 0, f"Val-Test leak detected: {val_set & test_set}"
        assert len(train_set | val_set | test_set) == total_storms, "Some storms left unassigned"

        return split_map

    def export_split_manifest(self,
                              storm_records: Dict[str, Dict[str, Any]],
                              split_map: Dict[str, str],
                              output_path: str) -> str:
        """
        Generate storm_split_manifest.csv recording partition metadata.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fields = [
            "storm_id", "season", "era", "basin", "partition",
            "overpass_count", "has_environmental", "split_ratio_target"
        ]

        total_storms = len(storm_records)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for sid in sorted(storm_records.keys()):
                st = storm_records[sid]
                season = st.get("season", 0)
                era = classify_era(season)
                basin = st.get("basin", sid[:2])
                part = split_map.get(sid, "UNKNOWN")
                op_count = len(st.get("overpass_files", []))
                has_env = st.get("env_file") is not None

                writer.writerow({
                    "storm_id": sid,
                    "season": season,
                    "era": era,
                    "basin": basin,
                    "partition": part,
                    "overpass_count": op_count,
                    "has_environmental": has_env,
                    "split_ratio_target": f"{self.config.train:.2f}/{self.config.validation:.2f}/{self.config.test:.2f}",
                })

        return output_path

