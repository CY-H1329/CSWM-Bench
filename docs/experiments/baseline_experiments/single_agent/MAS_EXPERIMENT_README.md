# MAS 실험 세팅 (Head → Perception → Reasoning)

Multi-Agent System 평가 실험 설정 및 실행 방법.

---

## 1. 아키텍처

```
Input: Query + 2D Image
       ↓
   Head-Agent      → Task 분류 (depth, relation, count, ...)
       ↓
   Perception Agent → 이미지에서 관련 정보 추출
       ↓
   Reasoning Agent  → 추론 후 최종 답변
       ↓
Output: Answer
```

---

## 2. 스크립트

| 스크립트 | 용도 |
|----------|------|
| `run_eval_mas.py` | 단일 조합 실행 (Head/Perception/Reasoning 지정) |
| `run_eval_mas_full.py` | **전체 평가**: Qwen3 X3, Sa2VA X3, LLaVA4D X3, 전체 데이터 |

---

## 3. run_eval_mas.py (단일 조합)

```bash
# Qwen3 4B로 Head/Perception/Reasoning 모두 사용
python run_eval_mas.py --benchmark stvqa7k --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b

# Sa2VA
python run_eval_mas.py --benchmark stvqa7k --head sa2va --perception sa2va --reasoning sa2va

# LLaVA4D
python run_eval_mas.py --benchmark stvqa7k --head llava4d --perception llava4d --reasoning llava4d

# 빠른 테스트 (100개 샘플)
python run_eval_mas.py --benchmark stvqa7k --head qwen3_4b --perception qwen3_4b --reasoning qwen3_4b --max_samples 100 --seed 123
```

**지원 모델**: `qwen3_4b`, `sa2va`, `llava4d`, `qwen`, `llava`, `gpt`, `gemini`

---

## 4. run_eval_mas_full.py (전체 평가)

**Qwen3 X3, Sa2VA X3, LLaVA4D X3** 세 가지 조합을 **전체 데이터**로 실행.

```bash
python run_eval_mas_full.py --benchmark stvqa7k
```

- `--benchmark`: stvqa7k, omni3d, cvbench, 3dsrbench
- `--seed`: 데이터 샘플링 시드 (기본: config `mas_seed`)

### 출력 구조

```
results/runs/<benchmark>/full_eval/<timestamp>/
├── qwen3_4b_qwen3_4b_qwen3_4b/
│   ├── results.json           # 전체 + category별 정확도
│   ├── by_category_summary.txt
│   ├── details.jsonl
│   └── step_outputs/
│       ├── sample_00000.txt    # Head, Perception, Reasoning 단계별 텍스트
│       ├── sample_00001.txt
│       └── ...
├── sa2va_sa2va_sa2va/
│   └── ...
├── llava4d_llava4d_llava4d/
│   └── ...
├── all_combinations_summary.txt  # 3개 조합 비교 + category별 표
└── config_snapshot.yaml
```

### step_outputs 형식 (각 샘플)

```
=== QUERY ===
질문 내용...

=== HEAD (Task Classification) ===
depth

=== PERCEPTION (Extracted Info) ===
이미지에서 추출한 정보...

=== REASONING (Final Answer) ===
최종 답변...

=== PRED ===
A

=== GT ===
A
```

---

## 5. config.yaml 주요 설정

```yaml
eval:
  mas_temperature: 0.0    # MAS용 (greedy, 재현성)
  mas_seed: 42           # 데이터 샘플링 시드
  max_new_tokens: 512

models:
  qwen3_4b:
    enabled: true
    model_id: "Qwen/Qwen3-VL-4B-Instruct"
  llava4d:
    enabled: true
    model_id: "llava-hf/llava-v1.6-mistral-7b-hf"  # LLaVA-4D 미출시 시 fallback
  sa2va:
    enabled: true
    model_id: "ByteDance/Sa2VA-4B"
    use_flash_attn: false   # PyTorch 호환 문제 시 false
```

---

## 6. Sa2VA 특이사항

- **flash-attn**: PyTorch 2.5 + flash-attn 2.8+ 호환 문제 시 `flash-attn==2.7.3` 사용
- **meta tensor**: `low_cpu_mem_usage=False`, `device_map=None` 등 적용됨 (`src/models/sa2va.py`)
- **tied weights**: transformers API 호환 패치 포함

---

## 7. 참고 문서

- [PROJECT_PLAN.md](./PROJECT_PLAN.md) - 실험 계획, Phase 1/2
- [MAS_PROMPTS.md](./MAS_PROMPTS.md) - Head/Perception/Reasoning 프롬프트
- [GPU_OPTIMIZATION.md](./GPU_OPTIMIZATION.md) - Flash Attention, TF32
