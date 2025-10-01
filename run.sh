#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단합니다.
set -e

echo "🚀 Few-Shot RAG 서비스 시작 스크립트 🚀"

# 1. Python 가상 환경 확인 및 생성
echo -e "\n--- 1. 가상 환경 설정 ---"
if [ ! -d ".venv" ]; then
    echo "'.venv' 가상 환경을 생성합니다."
    python3 -m venv .venv
else
    echo "'.venv' 가상 환경이 이미 존재합니다."
fi

# 2. 가상 환경 활성화
echo "가상 환경을 활성화합니다."
source .venv/bin/activate

# 3. 필수 라이브러리 설치
echo -e "\n--- 2. 필수 라이브러리 설치 ---"
echo "requirements.txt를 사용하여 라이브러리를 설치합니다..."
pip install -r requirements.txt

# 4. 벡터 데이터베이스 구축 (필요 시)
echo -e "\n--- 3. 벡터 데이터베이스 구축 ---"
if [ ! -d "data/chroma_db_korquad_full_context_rag" ]; then
    echo "벡터 DB가 존재하지 않습니다. 'db_setup.py'를 실행하여 새로 구축합니다."
    echo "이 작업은 데이터셋 크기에 따라 몇 분 정도 소요될 수 있습니다..."
    python3 db_setup.py
else
    echo "벡터 DB가 이미 존재하므로 구축 단계를 건너뜁니다."
fi

# 5. FastAPI 서버 실행
echo -e "\n--- 4. FastAPI 서버 실행 ---"
echo "Uvicorn 서버를 시작합니다. (http://127.0.0.1:8000)"
echo "API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요."
echo "서버를 중지하려면 Ctrl+C를 누르세요."
uvicorn app.main:app --host 0.0.0.0 --port 8000
