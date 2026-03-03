#!/usr/bin/env python3
"""
CV-Bench에서 Count(counting) 카테고리만 100개 추출하여 폴더에 저장.

저장 위치: data/frozen_benchmarks/cvbench_counting_100/
→ load_benchmark("cvbench_counting_100")로 로드 가능

Usage:
    python scripts/create_cvbench_counting_100.py
    python scripts/create_cvbench_counting_100.py --output_dir data/frozen_benchmarks/cvbench_counting_100 --n_samples 100
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src2.benchmarks.loaders import load_benchmark, FROZEN_BENCHMARK_DIR, BENCHMARK_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 폴더 (기본: data/frozen_benchmarks/cvbench_counting_100)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_frozen", action="store_true", default=True,
                        help="cvbench_400 frozen에서 로드 (없으면 HF)")
    parser.add_argument("--no_frozen", dest="use_frozen", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FROZEN_BENCHMARK_DIR / "cvbench_counting_100"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CV-Bench 로드 (Count만, 100개)
    ds = load_benchmark(
        "cvbench",
        max_samples=args.n_samples,
        category_filter=["Count"],
        seed=args.seed,
        use_frozen=args.use_frozen,
    )

    print(f"Count 샘플 {len(ds)}개 추출 완료")
    if len(ds) == 0:
        print("경고: Count 샘플이 없습니다. cvbench 데이터 확인 필요.")
        return 1

    # datasets 형식으로 저장 (load_from_disk 호환)
    ds.save_to_disk(str(out_dir))
    print(f"저장 완료: {out_dir}")
    print()
    print("사용법:")
    print('  load_benchmark("cvbench_counting_100")  # 또는 test에서 --benchmark cvbench_counting_100')
    return 0


if __name__ == "__main__":
    sys.exit(main())
