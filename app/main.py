from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="📚 Few-Shot RAG 기반 위키피디아 질의응답 서비스",
    description="KorQuAD 데이터를 기반으로 Few-Shot RAG를 사용하여 질문에 답변하는 API입니다.",
    version="1.0.0",
)

# CORS 미들웨어 추가
# 모든 출처에서의 요청을 허용합니다. 실제 프로덕션 환경에서는
# 보안을 위해 특정 도메인만 허용하는 것이 좋습니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# API 라우터 등록
app.include_router(endpoints.router, prefix="/rag", tags=["RAG"])

@app.get("/", tags=["Root"])
def read_root():
    """
    API 서버의 상태를 확인하는 기본 엔드포인트입니다.
    """
    return {"message": "Few-Shot RAG API 서버가 정상적으로 실행 중입니다."}
