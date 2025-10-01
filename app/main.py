from fastapi import FastAPI
from app.api import endpoints

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="📚 Few-Shot RAG 기반 위키피디아 질의응답 서비스",
    description="KorQuAD 데이터를 기반으로 Few-Shot RAG를 사용하여 질문에 답변하는 API입니다.",
    version="1.0.0",
)

# API 라우터 등록
app.include_router(endpoints.router, prefix="/rag", tags=["RAG"])

@app.get("/", tags=["Root"])
def read_root():
    """
    API 서버의 상태를 확인하는 기본 엔드포인트입니다.
    """
    return {"message": "Few-Shot RAG API 서버가 정상적으로 실행 중입니다."}
