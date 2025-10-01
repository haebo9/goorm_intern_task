from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.services.rag_service import initialize_rag_system

# --- 상태 관리 ---
# 서버가 요청을 처리할 준비가 되었는지 나타내는 플래그
is_ready = False

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="📚 Few-Shot RAG 기반 위키피디아 질의응답 서비스",
    description="KorQuAD 데이터를 기반으로 Few-Shot RAG를 수행하는 API입니다.",
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

@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 RAG 시스템을 초기화합니다.
    임베딩, 벡터DB, LLM을 미리 로드하여 첫 요청 시 지연을 방지합니다.
    """
    global is_ready
    print("--- FastAPI 애플리케이션 시작 ---")
    initialize_rag_system()
    is_ready = True  # 모든 초기화가 끝나면 플래그를 True로 설정
    print("--- 모든 모델 및 시스템 초기화 완료. 서비스 준비 완료. ---")

app.include_router(api_router, prefix="/rag", tags=["RAG"])

@app.get("/health", tags=["Status"])
def health_check():
    """
    서버가 모든 모델을 로드하고 요청을 처리할 준비가 되었는지 확인합니다.
    준비가 완료되면 HTTP 200과 함께 {"status": "ready"}를 반환합니다.
    아직 준비 중인 경우 HTTP 503을 반환합니다.
    """
    if is_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: Models are still loading.")

@app.get("/", tags=["Root"])
async def read_root():
    """
    API 서버의 상태를 확인하는 기본 엔드포인트입니다.
    """
    return {"message": "Few-Shot RAG API 서버가 정상적으로 실행 중입니다."}
