# Gemini 429 에러 & Google Cloud 크레딧

## 429 RESOURCE_EXHAUSTED 가 나는 이유

**돈을 안 내서가 아니라**, **할당량(rate limit) 초과** 때문입니다.

- **Google AI Studio API**(지금 쓰는 방식, API 키)에는 **무료 한도**가 있음:
  - **RPM** (분당 요청 수): 예: 10
  - **RPD** (하루 요청 수): 예: 250 (모델마다 다름)
- 692개를 연속으로 호출하면 **RPD나 RPM**을 금방 넘겨서 429가 납니다.

[할당량 확인](https://aistudio.google.com/usage?timeRange=last-28-days&tab=rate-limit)

---

## Google Cloud 크레딧($300 등)을 여기서 쓸 수 있나?

**지금 코드는 Google AI Studio API(API 키)를 쓰고 있어서, GCP 크레딧이 적용되지 않습니다.**

- **Google AI Studio API** (aistudio.google.com API 키)  
  → 무료 한도만 적용, **GCP 크레딧 사용 불가**
- **Vertex AI** (Google Cloud 프로젝트 + 결제 활성화)  
  → 여기서 쓰는 비용에 **GCP 크레딧 사용 가능**

그래서 **크레딧을 쓰려면** Gemini를 **Vertex AI**로 호출하도록 바꿔야 합니다 (엔드포인트·SDK가 다름).

---

## 할 수 있는 대응

1. **요청 사이에 간격 두기**  
   코드에 `delay_between_requests=6` 초 정도 넣어 두었습니다. (분당 10개 이하로 맞추기)

2. **한 번에 적게 돌리기**  
   ```bash
   python run_eval.py --models gemini --split val --max_samples 100
   ```  
   나중에 다시 돌려서 이어 붙이거나, 하루 한도가 리셋된 뒤 나머지를 돌리기.

3. **유료 사용으로 한도 올리기**  
   AI Studio에서 결제 연결하면 RPD 등 한도가 올라갑니다.

4. **Vertex AI로 전환해 크레딧 쓰기**  
   GCP 프로젝트 만들고, Vertex AI에서 Gemini 사용 설정한 뒤, 코드를 Vertex AI SDK/API로 바꾸면 **그때 쓰는 비용에 GCP 크레딧이 적용**됩니다.
