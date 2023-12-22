from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import torch

# 모델과 토크나이저 로드
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForQuestionAnswering.from_pretrained(
    "monologg/koelectra-base-v3-discriminator"
)

# 제공된 데이터
context = "일본의 법정 본인부담은 진료비의 일정률을 부담하는 정률제(coinsurance)방식이다. 본인부담분은 전체 진료비의 30%이며, 70~75세 미만 20%, 75세 이상 10%, 취학 전 아동은 20% 수준이다. 다만, 70세 이상에서도 일정소득 이상은 30%(일정소득 이상은 부부합산 연 소득 520만 엔 이상, 독신 383만 엔 이상)의 본인부담이 발생한다."
question = "자기가 부담하는 비용이 일본의 모든 진료비에서 차지하는 비율은 30%니"

# 질문과 문맥을 토크나이징 (최대 길이 512로 설정)
inputs = tokenizer.encode_plus(
    question,
    context,
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].tolist()[0]

# 모델로부터 답변 얻기
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits

# 가장 높은 점수를 가진 시작점과 끝점 찾기
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1

# 토큰을 문자열로 변환하여 답변 추출
extracted_answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)

print(extracted_answer)

# 답변이 문맥의 정보와 일치하는지 판단
if "30 %" in extracted_answer:
    final_answer = "Yes"
else:
    final_answer = "No"

# 최종 답변 출력
print("답변:", final_answer)
