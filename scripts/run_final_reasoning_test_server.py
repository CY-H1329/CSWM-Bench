#!/usr/bin/env python3
"""
Final Reasoning Agent (Full MAS v2) 테스트 — CV-Bench 100개.

Head → 3 Specialists → Final Reasoning Agent 전체 파이프라인 실행.
GPU 1개, prefetch로 이미지 병렬 로딩.

Jupyter에서 실행 (서버 경로에 맞게 PROJECT_ROOT 수정):
    PROJECT_ROOT = "/home/jovyan/CY/Spatial_MAS"  # 서버 경로
    import sys; sys.path.insert(0, PROJECT_ROOT)
    %run scripts/run_final_reasoning_test_server.py

또는 Python:
    python scripts/run_final_reasoning_test_server.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

# ========== 1. 설정 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 서버: PROJECT_ROOT = Path("/home/jovyan/CY/Spatial_MAS")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========== 2. 모델 로드 ==========
from run_eval_mas_v2 import build_runners
from test_final_reasoning_mas_v2 import run_mas_test

head_gen, spec_gen, reason_gen = build_runners(
    specialist_device="cuda",
    use_vlm_reasoning=True,  # Qwen3-VL-8B: 이미지 + SharedMemory로 Final Reasoning
    reasoning_vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
)


# ========== 3. CV-Bench 100개 ==========
print("\n" + "=" * 60)
print("Final Reasoning Agent — CV-Bench 100 samples")
print("=" * 60)
results_cv = run_mas_test(
    head_gen,
    spec_gen,
    reason_gen,
    benchmark="cvbench",
    max_samples=100,
    prefetch_workers=4,
    show_failures=20,
    use_vlm_reasoning=True,
)


# ========== 5. 요약 ==========
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"CV-Bench:    {results_cv.get('correct', 0)}/{results_cv.get('total', 0)} = {100*results_cv.get('accuracy', 0):.1f}%")
print("=" * 60)
