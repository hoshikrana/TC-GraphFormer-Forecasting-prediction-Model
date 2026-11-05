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

        # Stratified grouping by (basin, era)
        strata = defaultdict(list)
        for sid in all_sids:
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

