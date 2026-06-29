#!/usr/bin/env python3
"""Prefetch DFlash datasets via ModelScope into the global cache.

Data is **not** stored under the project tree. By default, files go to
``~/.cache/modelscope/hub/datasets/``. Override with ``--cache-dir``.

Usage::

    # List all datasets
    python scripts/download_datasets.py --list

    # Download everything
    python scripts/download_datasets.py

    # Download a subset
    python scripts/download_datasets.py --only gsm8k humaneval

    # Use a custom cache root (still outside the repo by default)
    python scripts/download_datasets.py --cache-dir ~/.cache/modelscope/hub
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from modelscope.hub.snapshot_download import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    modelscope_id: str
    description: str
    hf_equivalent: str
    usage: str  # train | eval
    available: bool = True


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        key="sharegpt",
        modelscope_id="swift/sharegpt",
        description="ShareGPT 多轮对话训练数据",
        hf_equivalent="lmsys/sharegpt_jsonl",
        usage="train",
    ),
    DatasetSpec(
        key="gsm8k",
        modelscope_id="AI-ModelScope/gsm8k",
        description="GSM8K 小学数学推理评测",
        hf_equivalent="openai/gsm8k",
        usage="eval",
    ),
    DatasetSpec(
        key="competition_math",
        modelscope_id="opencompass/competition_math",
        description="MATH 数学竞赛（math500 评测用）",
        hf_equivalent="hendrycks/competition_math",
        usage="eval",
    ),
    DatasetSpec(
        key="humaneval",
        modelscope_id="opencompass/humaneval",
        description="HumanEval 代码生成评测",
        hf_equivalent="openai/openai_humaneval",
        usage="eval",
    ),
    DatasetSpec(
        key="mbpp",
        modelscope_id="google-research-datasets/mbpp",
        description="MBPP Python 编程评测",
        hf_equivalent="google-research-datasets/mbpp",
        usage="eval",
    ),
    DatasetSpec(
        key="mt_bench",
        modelscope_id="",
        description="MT-Bench 对话质量评测",
        hf_equivalent="lmsys/fastchat (mt_bench_questions.jsonl)",
        usage="eval",
        available=False,
    ),
]

DATASET_BY_KEY = {d.key: d for d in DATASETS}


def _resolve_cache_dir(cache_dir: Optional[str]) -> Path:
    root = Path(os.path.expanduser(cache_dir or DEFAULT_CACHE_DIR)).resolve()
    project_root = PROJECT_ROOT.resolve()
    if root == project_root or project_root in root.parents:
        raise ValueError(
            f"cache_dir must not be inside the project ({project_root}); "
            f"got {root}"
        )
    return root


def _print_catalog() -> None:
    print("DFlash 所需数据集：\n")
    print(f"{'Key':<18} {'用途':<6} {'ModelScope ID':<40} HuggingFace 等价")
    print("-" * 110)
    for spec in DATASETS:
        ms_id = spec.modelscope_id or "(ModelScope 暂无)"
        print(f"{spec.key:<18} {spec.usage:<6} {ms_id:<40} {spec.hf_equivalent}")
        print(f"{'':18}        {spec.description}")
        if not spec.available:
            print(f"{'':18}        ⚠ 需从 HuggingFace / FastChat 手动获取")
        print()
    print(f"默认缓存目录: {DEFAULT_CACHE_DIR}")


def _select_datasets(keys: Optional[List[str]]) -> List[DatasetSpec]:
    if not keys:
        return [d for d in DATASETS if d.available]
    unknown = sorted(set(keys) - set(DATASET_BY_KEY))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}")
    selected = [DATASET_BY_KEY[k] for k in keys]
    missing = [d.key for d in selected if not d.available]
    if missing:
        raise ValueError(
            f"Dataset(s) not on ModelScope: {', '.join(missing)}. "
            "Use --list to see alternatives."
        )
    return selected


def _download_one(spec: DatasetSpec, cache_dir: Path) -> Path:
    print(f"\n>>> [{spec.key}] {spec.modelscope_id}")
    print(f"    {spec.description}")
    path = snapshot_download(
        spec.modelscope_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    root = Path(path)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    print(f"    cached: {path}")
    print(f"    files:  {len(files)}")
    return root


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch DFlash datasets via ModelScope.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List required datasets and exit.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="Download only the given dataset key(s), e.g. gsm8k humaneval.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"ModelScope cache root (default: {DEFAULT_CACHE_DIR}).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        _print_catalog()
        return 0

    try:
        cache_dir = _resolve_cache_dir(args.cache_dir)
        selected = _select_datasets(args.only)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    print(f"Downloading {len(selected)} dataset(s)...")

    failed: List[str] = []
    for spec in selected:
        try:
            _download_one(spec, cache_dir)
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(spec.key)

    print()
    if failed:
        print(f"Finished with errors: {', '.join(failed)}", file=sys.stderr)
        return 1

    print("All requested datasets are cached.")
    print("Training/eval code still uses HuggingFace dataset IDs in config YAML.")
    print("Point dataset_name to a local path only if you export from cache yourself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
