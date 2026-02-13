# GPU Optimization (MAS)

GPU 사용률을 높이기 위한 설정.

---

## config.yaml

```yaml
eval:
  use_flash_attn: true   # Flash Attention 2
  use_tf32: true         # TF32 (Ampere+, H100)
```

---

## Flash Attention 2

**설치 (H100 / CUDA 12.x):**
```bash
pip install flash-attn --no-build-isolation
```

- 설치 실패 시: `use_flash_attn: false` 로 설정하면 기본 attention 사용
- Qwen3-VL에서 메모리·속도 개선

---

## TF32

- Ampere 이상 GPU (A100, H100)에서 matmul 가속
- `use_tf32: true` 시 자동 활성화

---

## 참고

- **Batch size 1**: VLM inference는 보통 1개씩 처리. 4B 모델은 H100을 완전히 채우기 어려울 수 있음.
- **Sequential pipeline**: Head → Perception → Reasoning 3단계가 순차 실행되므로, GPU는 각 단계마다 한 번씩만 사용됨.
- **추가 옵션**: `torch.compile()` (PyTorch 2+) 또는 vLLM으로 배치 추론 시 고려 가능.
