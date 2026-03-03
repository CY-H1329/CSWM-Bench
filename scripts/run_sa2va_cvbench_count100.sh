#!/bin/bash
# SA2VA 평가: cvbench_400의 Count 카테고리 100개
# data/frozen_benchmarks/cvbench_400 사용 (Count만 100개)

cd "$(dirname "$0")/.."
python test_fixed_specialist_mas_v2.py \
  --specialist sa2va \
  --benchmark cvbench \
  --category_filter Count \
  --max_samples 100 \
  --device cuda
