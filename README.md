# 📚 Few-Shot RAG 기반 위키피디아 질의응답 서비스

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](https://github.com/your-repo)
[![Data](https://img.shields.io/badge/Data-KorQuAD%20v1-orange.svg)]()
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3--ko-yellowgreen.svg)]()
[![LLM](https://img.shields.io/badge/LLM-Kanana%20Instruct(4bit)-blue.svg)](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505)
[![Framework](https://img.shields.io/badge/Server-FastAPI-success.svg)](https://fastapi.tiangolo.com/)


---

## 1. 프로젝트 개요

본 서비스는 **KorQuAD (위키피디아 한국어 데이터)**를 활용하여 사용자의 질문에 대한 답변을 생성하는 **검색 증강 생성(RAG)** 서버입니다. FastAPI를 통해 REST API 형태로 제공됩니다.

### 🎯 핵심 목표

1.  질문에 대한 **정확한 답변** 생성.
2.  답변의 근거가 되는 **출처(문서) 제공**.
3.  출처에 없는 내용은 답변하지 않는 **질의 범위 제한** 구현.

---

## 2. 기술 스택 및 특징

| 분류 | 기술/모델 | 특징 |
| :--- | :--- | :--- |
| **LLM** | `kakaocorp/kanana-1.5-8b-instruct-2505` | Few-Shot 학습과 RAG를 활용하여 높은 정확도 확보. |
| **RAG 기법** | **Few-Shot RAG** | 검색된 문맥 외에 모범 질문/답변 예시를 제공하여 답변 품질 향상. |
| **최적화** | **4-bit 양자화** | LLM을 효율적으로 로드하여 GPU 메모리 사용 최소화. |
| **서버** | **FastAPI** | 비동기(Async) API 구조로 동시 접속 처리 능력 확보. |
| **데이터** | `squad_kor_v1` | KorQuAD 1.0 데이터셋 사용. |
| **임베딩** | `dragonkue/bge-m3-ko` | 한국어에 최적화된 고성능 임베딩 모델 사용. |

---

## 3. API 사용 방법

### 엔드포인트

`POST /rag/answer`

### 입력 (Request Body)

```json
{
  "question": "유엔이 설립된 연도는?",
  "k_fewshot": 5,
}
```

### 출력 (Response Body)
```json
{
  "answer": "1945년",
  "source_documents": [
    {
      "title": "유엔",
      "retrieved_question": "유엔은 언제 설립되었는가?",  
      "content_snippet": "유엔(UN)은 국제 연합(United Nations)의 약자로, 1945년 10월 24일에 설립된...",
      "is_fewshot": false 
    }
  ],
  "few_shot_examples_used": 3
}
```

## 4. 실행 가이드

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

### 벡터 데이터베이스 (DB) 구축
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
uvicorn main:app --host 0.0.0.0 --port 8000

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