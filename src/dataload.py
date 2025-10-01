## 데이터셋 불러오기
from datasets import load_dataset
dataset = load_dataset("squad_kor_v1")

train_data = dataset["train"]
val_data = dataset["validation"]

# 데이터 포멧 확인
print(dataset)