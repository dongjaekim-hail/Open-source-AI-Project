import os
import json

# 데이터가 있는 폴더 경로 (본인 경로에 맞게 수정)
folder_path = r""

# 모든 JSON 파일 읽어오기
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            all_data.extend(json_data["data"])

# 전처리를 위한 데이터셋 구성
preprocessed_data = []

for item in all_data:
    paragraphs = item.get("paragraphs", [])

    for paragraph in paragraphs:
        context = paragraph.get("context", "")
        qas = paragraph.get("qas", [])

        for qa in qas:
            question = qa.get("question", "")
            answer = qa.get("answer", {}).get("text", "")
            clue = qa.get("answer", {}).get("clue_text", "")

            # 전처리된 데이터셋에 추가
            preprocessed_data.append(
                {
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "clue_text": clue,
                }
            )

# 결과를 파일로 저장
output_file_path = os.path.join(folder_path, "preprocessed_qa_data.json")
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(preprocessed_data, output_file, ensure_ascii=False, indent=2)

print(f"Preprocessing complete. Preprocessed QA data saved to: {output_file_path}")
