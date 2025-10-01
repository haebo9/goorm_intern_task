# 📚 Few-Shot RAG 기반 위키피디아 질의응답 서비스

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](https://github.com/your-repo)
[![Data](https://img.shields.io/badge/Data-KorQuAD%20v1-orange.svg)](https://korquad.github.io/category/1.0_KOR.html)
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3--ko-yellowgreen.svg)](https://huggingface.co/dragonkue/BGE-m3-ko)
[![LLM](https://img.shields.io/badge/LLM-Kanana%20Instruct(4bit)-blue.svg)](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505)
[![Framework](https://img.shields.io/badge/Server-FastAPI-success.svg)](https://fastapi.tiangolo.com/)


---

## 1. 프로젝트 개요

본 서비스는 **KorQuAD (위키피디아 한국어 데이터)**를 활용하여 사용자의 질문에 대한 답변을 생성하는 **검색 증강 생성(RAG)** 서버입니다. FastAPI를 통해 REST API 형태로 제공됩니다.

### 🎯 핵심 목표

1.  질문에 대한 **정확한 답변** 생성.
2.  답변의 근거가 되는 **출처(문서) 제공**.
3.  출처에 없는 내용은 답변하지 않는 **질의 범위 제한** 구현.

### 📌 구현 전략
1.  **유사도 검색**: 사용자의 질문을 기반으로, 벡터 DB에서 가장 유사한 질문-답변 쌍을 `k_fewshot + 1`개 검색합니다.
2.  **역할 분담 (Few-Shot RAG)**:
    - **Top 1 (답변 근거)**: 가장 유사도가 높은 1개 문서는 LLM에게 답변의 직접적인 근거(Context)로 제공합니다.
    - **Top 2 ~ k+1 (Few-shot 예시)**: 나머지 `k`개 문서는 LLM에게 모범적인 질문/답변 형식을 보여주는 예시로 활용합니다.
3.  **답변 생성 및 반환**: LLM이 근거와 예시를 바탕으로 답변을 생성하면, 근거가 된 Top 1 문서를 출처 정보와 함께 반환합니다.
4. **환각(Hallucination) 억제**: 출처 기반 답변을 강제하는 프롬프트 설계를 통해 LLM의 환각 현상을 억제합니다.

---

## 2-1. 기술 스택 및 특징

| 분류 | 기술/모델 | 특징 |
| :--- | :--- | :--- |
| **LLM** | `kakaocorp/kanana-1.5-8b-instruct-2505` | Few-Shot 학습과 RAG를 활용하여 높은 정확도 확보. |
| **데이터** | `squad_kor_v1` | KorQuAD 1.0 데이터셋 사용. |
| **임베딩** | `dragonkue/bge-m3-ko` | 한국어에 최적화된 고성능 임베딩 모델 사용. |
| **RAG 기법** | `Few-Shot RAG` | 검색된 문맥 외에 모범 질문/답변 예시를 제공하여 답변 품질 향상. |
| **최적화** | `4-bit 양자화` | LLM을 효율적으로 로드하여 GPU 메모리 사용 최소화. |
| **서버** | `FastAPI` | 데이터 유효성 검사, 높은 성능을 제공하는 최신 웹 프레임워크. |


## 2-2. 데이터 명세
| 필드명 | 타입 | 데이터셋 필드 | 설명 |
| :--- | :--- | :--- | :--- |
| **title** | `string` | `title` | 출처 위키피디아 문서의 제목입니다. |
| **context** | `string` | `context` | 질문에 대한 답변을 포함하는 원문 컨텍스트 내용입니다. |
| **question** | `string` | `question` | 해당 context와 짝지어진 KorQuAD의 질문 필드입니다. |
| **id** | `string` | `id` | 문서의 고유 식별자입니다. |
| **answers** | `Dict` | `text`, `ànswer_start` | 질문의 정답, 정답이 포함된 문장 시작 인덱스 |
---
```json
## 데이터 예시
{
'id': '656656-0-0',
'title': '유엔',
'context': '유엔(UN)은 국제 연합(United Nations)의 약자로, 1945년 10월 24일에 설립된
국제 기구이다...',
'question': '유엔이 설립된 연도는?',
'answers': {'text': ['1945년'], 'answer_start': [54]}
}
```

## 3. API 사용 방법

### 엔드포인트

`POST /rag/answer`

### 입력 (Request Body)

```json
{
  "question": "유엔이 설립된 연도는?",
  "k_fewshot": 5
}
```

- `question` (str): 사용자 질문
- `k_fewshot` (int): RAG 검색 및 Few-shot 예시 생성에 사용할 문서의 수 (기본값: 3)

### 출력 (Response Body)
```json
{
  "answer": "1945년 10월 24일에 설립되었습니다.",
  "source_documents": [
    {
      "title": "유엔",
      "retrieved_question": "유엔은 언제 설립되었는가?",
      "content_snippet": "유엔(UN)은 국제 연합(United Nations)의 약자로, 1945년 10월 24일에 설립된...",
      "is_fewshot": false
    }
  ],
  "few_shot_examples_used": 5
}
```

## 4. 실행 가이드

### 하드웨어/소프트웨어 환경
| 하드웨어
- GPU : NVIDIA RTX 4090 (1x) 
- GPU 메모리 (VRAM) : 24 GB
- vCPU : 12
- 시스템 메모리 (RAM) : 32 GB 이상
- 디스크 공간 (SSD) : 60 GB 이상
- Container Disk(Disk usage) : 	60 GB
- Volume Disk (영구 저장소) : 80 GB

| 소프트웨어
- 리눅스 : Ubuntu 22.04
- pytorch : 2.1.0
- runpod : pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

### 환경 설정 및 라이브러리 설치
```bash
# 1. 가상 환경 생성
python3 -m venv .venv

# 2. 가상 환경 활성화
source .venv/bin/activate

# 3. 필수 라이브러리 설치
# (LLM/임베딩, 벡터DB, FastAPI 등)

# (선택 사항: requirements.txt 파일이 있다면 추가 설치)
pip install -r requirements.txt
```

### 벡터 데이터베이스 (Chroma DB) 구축
```bash
# 1. DB 구축 스크립트 실행
# (스크립트 파일명: rag_db_setup.py로 가정)
echo "RAG 벡터 DB 구축 시작..."
python3 db_setup.py

# 2. DB 구축 완료 확인
echo "--- DB 구축 결과 확인 ---"
ls -d chroma_db_korquad_full_context_rag/
echo "--------------------------"
```

### FastAPI 서버 실행
```bash
# 1. FastAPI 서버 실행
# LLM은 4-bit 양자화로 GPU에 로드됩니다.
# (파일 이름: main.py, FastAPI 인스턴스 이름: app)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. 서비스 확인 (서버 실행 로그 확인)
echo "--- RAG 서비스 실행 정보 ---"
echo "서비스가 8000 포트에서 실행 중입니다."
echo "API 문서 (Swagger UI): http://127.0.0.1:8000/docs"
echo "--------------------------"

# 3. (서버 종료 시) 가상 환경 비활성화
# CTRL + C 로 서버를 종료한 후 다음 명령어 실행
# deactivate
```

### + 자동 실행 방법

```bash
# 먼저 스크립트에 실행 권한을 부여
chmod +x run.sh

# 가상 환경 설정, 라이브러리 설치, 데이터베이스 구축, 서버 실행까지 모든 과정을 한 번에 처리
./run.sh
```

## 5. 실행 테스트 (구현 결과)
| 입력 예시
- k_fewshot : few-shot으로 사용할 데이터의 개수
```bash
curl -X POST "http://127.0.0.1:8000/rag/answer" \
-H "Content-Type: application/json" \
-d '{
  "question": "이순신 장군이 사망한 전쟁은 무엇인가?",
  "k_fewshot": 3
}'
```

| 출력 예시 
- answer : LLM RAG 출력 값
- source_documents : 참고한 데이터 k개 중 유사도가 가장 높은 top_1만 출력
    - title : 출처 위키피디아 제목
    - retrieved_question : 데이터의 질문(정답에 대한 질문)
    - content_snippet : 실제 정답 단어가 포함된 원문의 한 문장
    - is_fewshot=false : 해당 데이터가 직접적인 근거로 사용(true:단순 형식 참고용 : few-shot)

`(답변 신뢰도를 위해 Top_1을 최종 답변의 출처로 사용하고, 나머지 k-1개는 형식 참고용으로 사용)`

```json
{"answer":"노량해전","source_documents":[{"title":"이순신","retrieved_question":"이순신이 전사한 곳은?","content_snippet":"노량해협에 모여 있는 일본군을 공격하였고, 일본으로 건너갈 준비를 하고 있던 왜군 선단 500여 척 가운데 200여 척을 격파, 150여 척을 파손시켰다.","is_fewshot":false}],"few_shot_examples_used":3}
```