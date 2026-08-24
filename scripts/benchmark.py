"""Benchmark detector latency, throughput (FPS), and memory footprint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ultralytics import YOLO

from proctoring_cv.logging_utils import setup_logger
from evaluation.metrics import measure_throughput_fps


def benchmark_model(
    model_path: str = "yolo11n.yaml",
    img_size: int = 640,
    device: str = "cpu",
    warmup: int = 10,
    runs: int = 50,
) -> int:
    """Runs benchmarking on detector."""
    logger = setup_logger("benchmark")
    logger.info(f"Loading model '{model_path}' on device '{device}'...")

    try:
        model = YOLO(model_path, task="detect")
        dummy_frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        logger.info(f"Benchmarking {runs} iterations (imgsz={img_size})...")
        stats = measure_throughput_fps(model, dummy_frame, warmup_runs=warmup, test_runs=runs)

        logger.info("=== Benchmark Results ===")
        logger.info(f"Average Latency: {stats['average_latency_ms']} ms")
        logger.info(f"Throughput:      {stats['fps']} FPS")
        logger.info(f"Min Latency:     {stats['min_latency_ms']} ms")
        logger.info(f"Max Latency:     {stats['max_latency_ms']} ms")
        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO Detector")
    parser.add_argument("--model", type=str, default="yolo11n.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    sys.exit(benchmark_model(
        model_path=args.model,
        img_size=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
    ))


if __name__ == "__main__":
    main()
