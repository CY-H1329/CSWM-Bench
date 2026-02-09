#!/bin/bash
# 서버에 있는 Conda 환경 목록 + 현재(또는 지정) env 에서 필수 패키지 확인
# 사용법: cd ~/CY/Spatial_MAS && bash scripts/check_env.sh
#        bash scripts/check_env.sh my_env   (특정 env 확인)

set -e
cd "$(dirname "$0")/.."

echo "=== Conda 환경 목록 ==="
conda env list

if [ -n "$1" ]; then
  echo ""
  echo "=== '$1' 환경에서 패키지 확인 중 ==="
  eval "$(conda shell.bash hook)"
  conda activate "$1"
fi

echo ""
echo "=== 현재 환경 필수 패키지 확인 (Spatial_MAS 실행용) ==="
python - <<'PY'
import sys
ok = []
miss = []
for pkg, imp in [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("openai", "openai"),
    ("PIL", "pillow"),
    ("yaml", "pyyaml"),
]:
    try:
        __import__(imp)
        ok.append(pkg)
    except ImportError:
        miss.append(pkg)
print("설치됨:", ", ".join(ok) if ok else "(없음)")
if miss:
    print("없음:", ", ".join(miss))
    print("\n설치 명령 (현재 env):")
    print("  pip install transformers datasets huggingface_hub accelerate qwen-vl-utils pillow openai tqdm pyyaml pandas numpy")
else:
    print("\n이 환경에서 실행 가능: python run_eval.py --models qwen llava gpt --split val")
if "torch" in ok:
    try:
        import torch
        cuda = torch.cuda.is_available()
        print("CUDA 사용 가능:", cuda)
    except Exception:
        pass
PY
