"""
Pipeline Runner module for TC PRIMED.
Coordinates data discovery, validation, splitting, incremental processing,
sample building, and report generation with full resume support and disk space safety.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional
import netCDF4 as nc

from .config import PipelineConfig
from .inventory import DatasetInventory
from .netcdf_reader import open_netcdf
from .metadata_extractor import MetadataExtractor
from .modality_processors import ModalityProcessors
from .environmental_matcher import EnvironmentalMatcher
from .quality_control import QualityController
from .storm_splitter import StormSplitter
from .sample_builder import SampleBuilder
from .storage_writer import StorageWriter
from .reporting import PipelineReporter


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tcprimed_pipeline")


class PipelineRunner:
    """Master coordinator for TC PRIMED data processing."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.inventory = DatasetInventory(config)
        self.qc = QualityController(config.quality_control)
        self.splitter = StormSplitter(config.splitting)
        self.sample_builder = SampleBuilder(config.targets)
        self.storage_writer = StorageWriter(config.output)
        self.env_matcher = EnvironmentalMatcher(config.quality_control.enforce_causal_environmental)

    def run_inventory(self) -> Dict[str, Any]:
        """Discover and catalogue dataset files."""
        logger.info("Starting Dataset Inventory scan...")
        self.config.ensure_directories()
        self.inventory.scan()

        f_man, obs_man = self.inventory.generate_manifests(self.config.output.reports_dir)
        logger.info(f"Generated file manifest: {f_man}")
        logger.info(f"Generated observation manifest: {obs_man}")

        total_files = len(self.inventory.files)
        total_storms = len(self.inventory.storms)
        overpass_count = sum(len(s["overpass_files"]) for s in self.inventory.storms.values())
        env_count = sum(1 for s in self.inventory.storms.values() if s["env_file"] is not None)

        instruments = set(f["instrument"] for f in self.inventory.files if f["file_type"] == "overpass")
        seasons = [f["season"] for f in self.inventory.files if f["season"] > 0]
        season_range = f"{min(seasons)}-{max(seasons)}" if seasons else "Unknown"

        logger.info(f"Discovered {total_files} total files across {total_storms} storms.")
        logger.info(f"Overpasses: {overpass_count}, Environmental files: {env_count}")
        logger.info(f"Sensors found: {sorted(list(instruments))}")

        return {
            "total_files": total_files,
            "overpass_files": overpass_count,
            "env_files": env_count,
            "total_storms": total_storms,
            "instruments": list(instruments),
            "season_range": season_range,
        }

    def run_split(self) -> Dict[str, Any]:
        """Perform storm-wise train/val/test splitting."""
        logger.info("Executing Storm-Wise Partitioning...")
        if not self.inventory.storms:
            self.inventory.scan()

        split_map = self.splitter.split_storms(self.inventory.storms)
        split_manifest_path = os.path.join(self.config.output.reports_dir, "storm_split_manifest.csv")
        self.splitter.export_split_manifest(self.inventory.storms, split_map, split_manifest_path)
        logger.info(f"Exported storm split manifest: {split_manifest_path}")

        # Compute summary
        counts = {"TRAIN": 0, "VAL": 0, "TEST": 0}
        obs_counts = {"TRAIN": 0, "VAL": 0, "TEST": 0}
        for sid, part in split_map.items():
            counts[part] += 1
            obs_counts[part] += len(self.inventory.storms[sid]["overpass_files"])

        tot_s = len(split_map)
        logger.info(f"Split results: Train={counts['TRAIN']} storms ({100*counts['TRAIN']/tot_s:.1f}%), "
                    f"Val={counts['VAL']} storms ({100*counts['VAL']/tot_s:.1f}%), "
                    f"Test={counts['TEST']} storms ({100*counts['TEST']/tot_s:.1f}%)")

        return {
            "split_map": split_map,
            "train_storms": counts["TRAIN"],
            "train_storms_pct": 100.0 * counts["TRAIN"] / tot_s if tot_s > 0 else 0.0,
            "val_storms": counts["VAL"],
            "val_storms_pct": 100.0 * counts["VAL"] / tot_s if tot_s > 0 else 0.0,
            "test_storms": counts["TEST"],
            "test_storms_pct": 100.0 * counts["TEST"] / tot_s if tot_s > 0 else 0.0,
            "train_obs": obs_counts["TRAIN"],
            "val_obs": obs_counts["VAL"],
            "test_obs": obs_counts["TEST"],
        }

    def run_process(self, resume: Optional[bool] = None) -> Dict[str, Any]:
        """
        Process raw NetCDF files into standardized per-storm NetCDF-4/HDF5 containers.
        Supports resume and checks disk space safely.
        """
        do_resume = self.config.processing.resume if resume is None else resume
        self.config.check_disk_space()

        if not self.inventory.storms:
            self.inventory.scan()

        split_map = self.splitter.split_storms(self.inventory.storms)
        total_storms = len(self.inventory.storms)
        logger.info(f"Starting processing across {total_storms} storms (resume={do_resume})...")

        t_start = time.time()
        storms_processed = 0
        storms_skipped = 0
        overpasses_processed = 0
        total_raw_bytes = 0

        for sid, sdata in sorted(self.inventory.storms.items()):
            season = sdata["season"]
            basin = sdata["basin"]
            partition = split_map.get(sid, "TRAIN")

            # Check resume
            if do_resume and self.storage_writer.is_storm_processed(season, basin, sid):
                logger.info(f"Storm {sid} already processed. Re-indexing from container for reports...")
                storms_skipped += 1
                cpath = self.storage_writer.get_storm_container_path(season, basin, sid)
                self.storage_writer.processed_storms[sid] = cpath
                self.storage_writer.output_bytes_written += os.path.getsize(cpath)
                for op in sdata["overpass_files"]:
                    total_raw_bytes += op["file_size_bytes"]
                if sdata["env_file"]:
                    total_raw_bytes += sdata["env_file"]["file_size_bytes"]

                # Read observation index and targets from processed container
                with open_netcdf(cpath, "r") as ds_c:
                    # Best track for future positions
                    bt = {"times": [], "lats": [], "lons": []}
                    if "storm_metadata" in ds_c.groups:
                        g_sm = ds_c.groups["storm_metadata"]
                        if "time" in g_sm.variables:
                            bt["times"] = g_sm.variables["time"][:]
                        if "storm_latitude" in g_sm.variables:
                            bt["lats"] = g_sm.variables["storm_latitude"][:]
                        if "storm_longitude" in g_sm.variables:
                            bt["lons"] = g_sm.variables["storm_longitude"][:]

                    g_ops = ds_c.groups.get("overpasses")
                    if g_ops:
                        for obs_gname, g_obs in g_ops.groups.items():
                            obs_id = getattr(g_obs, "observation_id", obs_gname)
                            inst = getattr(g_obs, "instrument", "")
                            plat = getattr(g_obs, "platform", "")
                            t_ep = int(getattr(g_obs, "time_epoch", 0))
                            t_utc = getattr(g_obs, "time_utc", "")
                            mod_flags = {
                                "has_pmw": bool(getattr(g_obs, "has_pmw", False)),
                                "has_ir": bool(getattr(g_obs, "has_ir", False)),
                                "has_gprof": bool(getattr(g_obs, "has_gprof", False)),
                                "has_radar": bool(getattr(g_obs, "has_radar", False)),
                            }
                            targets = {}
                            if "targets" in g_obs.groups:
                                g_tgt = g_obs.groups["targets"]
                                for vname in g_tgt.variables:
                                    targets[vname] = float(g_tgt.variables[vname][:])
                                for aname in g_tgt.ncattrs():
                                    targets[aname] = getattr(g_tgt, aname)

                            future_track = MetadataExtractor.query_future_track_coordinates(
                                bt, t_ep, self.config.targets.horizons_hours
                            )

                            env_match = {
                                "has_match": bool(getattr(g_obs, "has_causal_env", False)),
                                "time_diff_hours": getattr(g_obs.groups.get("environmental_synoptic", {}), "time_diff_hours", None) if "environmental_synoptic" in g_obs.groups else None,
                            }
                            storage_info = {
                                "container_path": cpath,
                                "group_path": f"/overpasses/{obs_gname}",
                            }
                            op_meta = {
                                "instrument": inst,
                                "platform": plat,
                                "time_epoch": t_ep,
                                "time_utc": t_utc,
                            }
                            self.sample_builder.add_sample(
                                observation_id=obs_id,
                                storm_id=sid,
                                partition=partition,
                                season=season,
                                basin=basin,
                                overpass_meta=op_meta,
                                targets=targets,
                                future_track=future_track,
                                modality_flags=mod_flags,
                                env_match=env_match,
                                storage_info=storage_info,
                            )
                            # Record QC audit for resumed observation
                            self.qc.validate_overpass(
                                filepath=f"{cpath}#{obs_gname}",
                                extracted_metadata=op_meta,
                                pmw_data={"is_available": mod_flags["has_pmw"]},
                                ir_data={"is_available": mod_flags["has_ir"]},
                                targets=targets,
                            )
                            overpasses_processed += 1
                continue

            # Read Environmental Best Track
            env_rec = sdata["env_file"]
            best_track = {"times": [], "lats": [], "lons": [], "intensities": [], "pressures": []}
            if env_rec and os.path.exists(env_rec["file_path"]):
                total_raw_bytes += env_rec["file_size_bytes"]
                self.env_matcher.register_storm_environmental_file(sid, env_rec["file_path"])
                with open_netcdf(env_rec["file_path"], "r") as ds_env:
                    best_track = MetadataExtractor.extract_environmental_best_track(ds_env)

            # Process all overpasses for this storm
            storm_observations = []
            for op in sdata["overpass_files"]:
                op_path = op["file_path"]
                total_raw_bytes += op["file_size_bytes"]

                if not os.path.exists(op_path):
                    continue

                with open_netcdf(op_path, "r") as ds_op:
                    # 1. Metadata
                    op_meta = MetadataExtractor.extract_overpass_metadata(ds_op)
                    t_epoch = op_meta["time_epoch"] or 0
                    targets = MetadataExtractor.extract_overpass_targets(ds_op)

                    # 2. Future track coordinates from Best Track
                    future_track = MetadataExtractor.query_future_track_coordinates(
                        best_track, t_epoch, self.config.targets.horizons_hours
                    )

                    # 3. Modalities
                    pmw_data = ModalityProcessors.process_passive_microwave(ds_op)
                    ir_data = ModalityProcessors.process_infrared(ds_op)
                    gprof_data = ModalityProcessors.process_gprof(ds_op)
                    radar_data = ModalityProcessors.process_radar(ds_op)

                    # 4. Environmental match
                    env_match = self.env_matcher.match_overpass(sid, t_epoch)
                    env_slice_data = {}
                    if env_match.get("has_match") and env_rec:
                        matched_idx = env_match["matched_index"]
                        with open_netcdf(env_rec["file_path"], "r") as ds_env:
                            env_slice_data = ModalityProcessors.process_environmental_slice(ds_env, matched_idx)

                    # 5. Quality Control validation
                    is_valid, issues = self.qc.validate_overpass(
                        op_path, op_meta, pmw_data, ir_data, targets
                    )

                    # Modality flags
                    mod_flags = {
                        "has_pmw": pmw_data.get("is_available", False),
                        "has_ir": ir_data.get("is_available", False),
                        "has_gprof": gprof_data.get("is_available", False),
                        "has_radar": radar_data.get("is_available", False),
                    }

                    obs_record = {
                        "observation_id": f"{sid}_{op['instrument']}_{op['platform']}_{op['timestamp_str']}",
                        "timestamp_str": op["timestamp_str"],
                        "timestamp_epoch": t_epoch,
                        "timestamp_utc": op_meta["time_utc"],
                        "instrument": op["instrument"],
                        "platform": op["platform"],
                        "has_pmw": mod_flags["has_pmw"],
                        "has_ir": mod_flags["has_ir"],
                        "has_gprof": mod_flags["has_gprof"],
                        "has_radar": mod_flags["has_radar"],
                        "has_causal_env": env_match.get("has_match", False),
                        "env_time_diff_hours": env_match.get("time_diff_hours"),
                        "env_is_causal": not env_match.get("is_future_leakage", False),
                        "targets": targets,
                        "pmw_data": pmw_data,
                        "ir_data": ir_data,
                        "gprof_data": gprof_data,
                        "radar_data": radar_data,
                        "env_slice_data": env_slice_data,
                    }
                    storm_observations.append(obs_record)

                    # Index sample
                    dest_cpath = self.storage_writer.get_storm_container_path(season, basin, sid)
                    storage_info = {
                        "container_path": dest_cpath,
                        "group_path": f"/overpasses/obs_{op['timestamp_str']}_{op['instrument']}",
                    }
                    self.sample_builder.add_sample(
                        observation_id=obs_record["observation_id"],
                        storm_id=sid,
                        partition=partition,
                        season=season,
                        basin=basin,
                        overpass_meta=op_meta,
                        targets=targets,
                        future_track=future_track,
                        modality_flags=mod_flags,
                        env_match=env_match,
                        storage_info=storage_info,
                    )
                    overpasses_processed += 1

            # Write storm container atomically
            if storm_observations:
                self.storage_writer.write_storm_container(
                    storm_id=sid,
                    season=season,
                    basin=basin,
                    partition=partition,
                    best_track=best_track,
                    observations=storm_observations,
                )
                storms_processed += 1

            if storms_processed % 5 == 0:
                logger.info(f"Progress: {storms_processed + storms_skipped}/{total_storms} storms processed/resumed.")

        elapsed = time.time() - t_start
        logger.info(f"Processing complete in {elapsed:.2f}s: {storms_processed} new storms written, "
                    f"{storms_skipped} skipped, {overpasses_processed} observations indexed.")

        return {
            "elapsed_seconds": round(elapsed, 2),
            "storms_processed": storms_processed,
            "storms_skipped": storms_skipped,
            "overpasses_processed": overpasses_processed,
            "total_raw_bytes": total_raw_bytes,
        }

    def run_all(self, resume: bool = True) -> Dict[str, Any]:
        """Execute end-to-end pipeline and compile all reports."""
        logger.info("=== EXECUTING FULL TC PRIMED PIPELINE ===")
        rep_dir = self.config.output.reports_dir

        inv_summary = self.run_inventory()
        split_summary = self.run_split()
        proc_summary = self.run_process(resume=resume)

        # Export QC Reports
        qc_json, qc_failures = self.qc.export_qc_reports(rep_dir)
        with open(qc_json, "r", encoding="utf-8") as f:
            qc_data = json.load(f)

        # Export Sample Index & Target Availability
        sample_index_path = os.path.join(rep_dir, "sample_index.csv")
        self.sample_builder.export_sample_index(sample_index_path)

        target_avail_path = os.path.join(rep_dir, "task_target_availability.csv")
        self.sample_builder.export_task_target_availability(target_avail_path)

        # Export Storage Benchmark
        benchmark_path = os.path.join(rep_dir, "storage_benchmark.json")
        self.storage_writer.export_storage_benchmark(proc_summary["total_raw_bytes"], benchmark_path)
        with open(benchmark_path, "r", encoding="utf-8") as f:
            storage_data = json.load(f)

        # Export Modality Availability
        mod_avail_path = os.path.join(rep_dir, "modality_availability.csv")
        PipelineReporter.export_modality_availability(self.sample_builder.samples, mod_avail_path)

        # Export Unresolved Scientific Issues
        unresolved_path = os.path.join(rep_dir, "unresolved_scientific_issues.md")
        PipelineReporter.export_unresolved_scientific_issues(unresolved_path)

        # Compile Target Summary for Master Report
        target_rows = []
        for s_task in [("Current Intensity (V_max)", "intensity_vmax_kt", "Continuous 1-min wind"),
                       ("Current Min Pressure (P_min)", "central_min_pressure_hpa", "Historical agency omission"),
                       ("Intensity Change (+6h)", "intensity_change_6h_kt", "Storm dissipation / tracking termination"),
                       ("Intensity Change (+12h)", "intensity_change_12h_kt", "Storm dissipation / tracking termination"),
                       ("Intensity Change (+24h)", "intensity_change_24h_kt", "Storm dissipation / tracking termination"),
                       ("Rapid Intensification (>=30kt/24h)", "is_rapid_intensification_24h", "24-hour intensification >= 30 kt"),
                       ("Future Track (+6h)", "future_lat_6h", "Sequential Best Track extraction"),
                       ("Future Track (+24h)", "future_lat_24h", "Sequential Best Track extraction")]:
            v_cnt = sum(1 for s in self.sample_builder.samples if s[s_task[1]] is not None)
            tot_cnt = len(self.sample_builder.samples)
            pct = round(100.0 * v_cnt / tot_cnt, 1) if tot_cnt > 0 else 0.0
            target_rows.append({"task": s_task[0], "valid": v_cnt, "total": tot_cnt, "pct": pct, "reason": s_task[2]})

        # Export Phase 2A Master Report
        phase2a_report_path = os.path.join(rep_dir, "phase2a_pipeline_implementation_report.md")
        PipelineReporter.generate_phase2a_report(
            config=self.config,
            inventory_summary=inv_summary,
            split_summary=split_summary,
            qc_summary=qc_data,
            storage_summary={
                "raw_pilot_gb": storage_data["pilot_benchmark"]["total_raw_pilot_gb"],
                "processed_pilot_gb": storage_data["pilot_benchmark"]["total_processed_gb"],
                "compression_ratio": storage_data["pilot_benchmark"]["compression_ratio_achieved"],
                "projected_storage_gb": storage_data["future_500gb_deployment_estimate"]["projected_total_storage_needed_gb"],
            },
            target_summary={"rows": target_rows},
            output_path=phase2a_report_path,
        )

        logger.info(f"=== FULL PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info(f"Master report saved to: {phase2a_report_path}")

        return {
            "inventory": inv_summary,
            "split": split_summary,
            "processing": proc_summary,
            "qc": qc_data,
            "storage": storage_data,
            "report_path": phase2a_report_path,
        }

