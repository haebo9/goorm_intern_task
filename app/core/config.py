from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """
    애플리케이션 설정을 관리하는 클래스.
    .env 파일 또는 환경 변수에서 설정을 로드합니다.
    """
    # LLM 모델 설정
    LLM_MODEL_ID: str = "kakaocorp/kanana-1.5-8b-instruct-2505"

    # 임베딩 모델 설정
    EMBED_MODEL_ID: str = "dragonkue/bge-m3-ko"

    # 벡터 DB 경로
    CHROMA_DB_PATH: str = "data/chroma_db_korquad_full_context_rag"

    # Few-Shot RAG에서 사용할 예시의 수
    DEFAULT_K_FEWSHOT: int = 3

    # 모델 로드 장치 설정 (CUDA 사용 가능 여부에 따라 자동 설정)
    DEVICE_TYPE: str = "auto"

    # LLM 답변 생성 시 최대 토큰 길이
    MAX_NEW_TOKENS: int = 512

    class Config:
        # .env 파일을 읽어 환경 변수로 사용
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 설정 객체 생성
settings = Settings()
