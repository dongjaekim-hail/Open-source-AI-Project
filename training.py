from transformers import (
    ElectraTokenizer,
    ElectraForQuestionAnswering,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch


# 데이터셋 로드 함수 정의
def load_data(train_path, validation_path):
    return load_dataset(
        "json", data_files={"training": train_path, "validation": validation_path}
    )


# JSON 파일 경로
train_json_path = "training.json"
validation_json_path = "validation.json"

# 데이터셋 로드
dataset = load_data(train_json_path, validation_json_path)

# 토크나이저 및 모델 초기화 (한국어 모델로 변경)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForQuestionAnswering.from_pretrained(
    "monologg/koelectra-base-v3-discriminator"
)


# 데이터셋 전처리 함수 (한국어 대응으로 수정)
def preprocess_data(examples):
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding=True,
        max_length=512,
        stride=128,
        return_tensors="pt",
    )

    start_positions = []
    end_positions = []
    for i in range(len(examples["question"])):
        # 한국어 답변 위치 찾기
        answer = examples["answer"][i]
        if answer == "Yes":
            answer_str = "예"
        else:
            answer_str = "아니요"
        answer_str = examples["clue_text"][i]

        # context 내에서 해당 문자열의 위치 찾기
        start_position = examples["context"][i].find(answer_str)
        end_position = start_position + len(answer_str) - 1

        start_positions.append(start_position)
        end_positions.append(end_position)

    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions

    # for those who start_positions == -1, remove the data from tokenized_inputs
    # tokenized_inputs = {key: [value[i] for i in range(len(start_positions)) if start_positions[i] != -1] for key, value in tokenized_inputs.items()}

    return tokenized_inputs


# 전처리 함수 적용
encoded_dataset = dataset.map(preprocess_data, batched=True)

# 트레이닝 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)


# 커스텀 손실 함수 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# 트레이너 초기화 및 트레이닝 시작
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["training"],
    eval_dataset=encoded_dataset["validation"],
)

trainer.train()

# get inference from the trained model
inferred_output=trainer.predict(encoded_dataset["validation"])

import numpy as np

# Assuming inferred_output is the result from trainer.predict
predictions = inferred_output.predictions

# Convert logits to probabilities (softmax)
start_logits, end_logits = predictions
start_probs = torch.softmax(torch.tensor(start_logits), dim=-1).numpy()
end_probs = torch.softmax(torch.tensor(end_logits), dim=-1).numpy()

# Find the token positions with the highest start and end probabilities
start_indexes = np.argmax(start_probs, axis=1)
end_indexes = np.argmax(end_probs, axis=1)

answers = []

for i, (start_index, end_index) in enumerate(zip(start_indexes, end_indexes)):
    # Map token positions to character positions in the original context
    context = encoded_dataset["validation"]["context"][i]
    # Extract the answer text
    answer = context[start_index:end_index+ 1]
    answers.append(answer)

clue_text = encoded_dataset["validation"]["clue_text"]
answer_percent = []

# comparison between the predicted answer and the clue_text, just showing
for count, answer in enumerate(answers):
    print("Predicted answer: \n")
    print(answer)
    print("Clue text: \n")
    print(clue_text[count])
    # calculate the overlap score
    overlap_score = len(set(answer) & set(clue_text[count])) / len(set(answer) | set(clue_text[count]))
    print("\t\tOverlap score: ", overlap_score)
    print("=====================================")
    answer_percent.append(overlap_score)


# final score
print("Final score: ", np.mean(answer_percent))
