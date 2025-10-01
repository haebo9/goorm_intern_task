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
echo -e "\n--- 3. 필수 라이브러리 설치 ---"
echo "requirements.txt를 사용하여 라이브러리를 설치합니다..."
pip install -r requirements.txt

# 4. 벡터 데이터베이스 확인 및 생성
echo -e "\n--- 4. 벡터 데이터베이스 확인 및 생성 ---"
if [ ! -d "data/chroma_db_korquad_full_context_rag" ]; then
    echo "벡터 DB가 존재하지 않습니다. 'db_setup.py'를 실행하여 새로 구축합니다."
    echo "이 작업은 데이터셋 크기에 따라 몇 분 정도 소요될 수 있습니다..."
    python db_setup.py
else
    echo "벡터 DB가 이미 존재하므로 구축 단계를 건너뜁니다."
fi

# 5. FastAPI 서버 실행 및 확인
echo -e "\n--- 5. FastAPI 서버 시작 및 모델 로드 확인 ---"
echo "Uvicorn 서버를 백그라운드에서 시작합니다. 주소: http://127.0.0.1:8000"
# --reload 옵션은 개발 중에 유용하며, 파일 변경 시 서버를 자동으로 재시작합니다.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!

# 서버가 모델 로드를 완료하고 준비될 때까지 대기 (최대 5분)
echo "서버가 시작되고 모델을 로드하는 중입니다. 이 과정은 몇 분 정도 소요될 수 있습니다..."
for i in {1..60}; do
    # /health 엔드포인트로 서버 준비 상태 확인
    response_code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
    if [ "$response_code" -eq 200 ]; then
        echo "✅ 서버가 성공적으로 시작되고 모든 모델을 로드했습니다."
        break
    fi
    echo "($i/60) 아직 서버가 준비되지 않았습니다. 5초 후 재시도합니다. (HTTP 상태: $response_code)"
    sleep 5
done

# 최종 서버 상태 확인
response_code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
if [ "$response_code" -ne 200 ]; then
    echo "❌ 서버가 5분 내에 정상적으로 준비되지 않았습니다. 로그를 확인해주세요."
    kill $UVICORN_PID
    exit 1
fi

echo -e "\n🎉 모든 준비가 완료되었습니다! 🎉"
echo "API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요."
echo "서버를 종료하려면 'Ctrl + C'를 누르거나 다음 명령어를 실행하세요: kill $UVICORN_PID"

# 백그라운드에서 실행된 uvicorn 프로세스가 종료될 때까지 대기
wait $UVICORN_PID