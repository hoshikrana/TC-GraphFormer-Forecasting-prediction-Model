"""
CLI entrypoint for tcprimed_pipeline.
Supports subcommands: inventory, validate, split, process, build-samples, run.
"""

import sys
import argparse
import logging
from typing import List

from .config import PipelineConfig
from .pipeline_runner import PipelineRunner


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        prog="python -m tcprimed_pipeline",
        description="TC PRIMED Modular Preprocessing Pipeline",
    )
    parser.add_argument(
        "subcommand",
        choices=["inventory", "validate", "split", "process", "build-samples", "run"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/pilot.yaml",
        help="Path to pipeline YAML configuration file (default: configs/pilot.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume processing and skip already-completed storm containers",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count from configuration",
    )

    args = parser.parse_args(argv)

    # Load configuration
    config = PipelineConfig.from_yaml(args.config)
    if args.workers is not None:
        config.processing.workers = args.workers
    if args.resume is not None:
        config.processing.resume = args.resume

    runner = PipelineRunner(config)

    if args.subcommand == "inventory":
        runner.run_inventory()
    elif args.subcommand == "split":
        runner.run_split()
    elif args.subcommand == "process":
        runner.run_process(resume=config.processing.resume)
    elif args.subcommand == "validate":
        runner.run_inventory()
        runner.run_process(resume=config.processing.resume)
        runner.qc.export_qc_reports(config.output.reports_dir)
    elif args.subcommand == "build-samples":
        runner.run_inventory()
        runner.run_split()
        runner.run_process(resume=config.processing.resume)
        runner.sample_builder.export_sample_index(f"{config.output.reports_dir}/sample_index.csv")
        runner.sample_builder.export_task_target_availability(f"{config.output.reports_dir}/task_target_availability.csv")
    elif args.subcommand == "run":
        runner.run_all(resume=config.processing.resume)


if __name__ == "__main__":
    main()

