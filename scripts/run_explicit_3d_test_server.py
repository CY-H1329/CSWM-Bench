#!/usr/bin/env python3
"""
Server에서 explicit_3d_representation + Qwen3 테스트 실행.

3DSRBench 50개, CV-Bench 50개 각각 실행 후 정답률 확인.

Jupyter에서 실행 (서버 경로에 맞게 PROJECT_ROOT 수정):
    PROJECT_ROOT = "/home/jovyan/CY/Spatial_MAS"  # 서버 경로
    import sys; sys.path.insert(0, PROJECT_ROOT)
    %run scripts/run_explicit_3d_test_server.py

또는 Python:
    python scripts/run_explicit_3d_test_server.py
"""
import sys
import subprocess
from pathlib import Path

# 서버: "/home/jovyan/CY/Spatial_MAS" 로 변경 가능
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Git pull
subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "origin", "main"], check=True)

from src2.models.qwen3 import Qwen3Runner
from test_specialist_explicit_3d import run_specialist_test

runner = Qwen3Runner(device="cuda")

print("\n" + "=" * 60)
print("1. CV-Bench 50 samples")
print("=" * 60)
results_cv = run_specialist_test(
    runner,
    benchmark="cvbench",
    max_samples=50,
    show_failures=10,
)

print("\n" + "=" * 60)
print("2. 3DSRBench 50 samples")
print("=" * 60)
results_3d = run_specialist_test(
    runner,
    benchmark="3dsrbench",
    max_samples=50,
    show_failures=10,
)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"CV-Bench:    {results_cv.get('correct', 0)}/{results_cv.get('total', 0)} = {100*results_cv.get('accuracy', 0):.1f}%")
print(f"3DSRBench:  {results_3d.get('correct', 0)}/{results_3d.get('total', 0)} = {100*results_3d.get('accuracy', 0):.1f}%")
print("=" * 60)
