# 프로젝트 구조: GOORM_INTERN_TASK

프로젝트는 FastAPI 모범 사례를 따라 핵심 로직, API 엔드포인트, 설정을 명확하게 분리하여 관리하고 있습니다.

```bash
GOORM_INTERN_TASK/
├── .venv/                         # Python 가상 환경 (Git .gitignore에 추가)
├── app/                           # (핵심) FastAPI 애플리케이션 모듈
│   ├── api/                       #    API 라우터 정의
│   │   └── endpoints.py           #        -> RAG 질의응답 엔드포인트 (/rag/answer)
│   ├── core/                      #    핵심 설정
│   │   └── config.py              #        -> .env 파일 로드 및 애플리케이션 설정 관리
│   ├── models/                    #    Pydantic 데이터 모델
│   │   └── schemas.py             #        -> API 입출력 데이터 형식 정의
│   ├── services/                  #    비즈니스 로직
│   │   └── rag_service.py         #        -> Few-Shot RAG 추론 로직 구현
│   └── main.py                    #    FastAPI 앱 인스턴스 생성 및 라우터 연결
├── data/                          # 데이터 저장소 (DB 등)
│   └── chroma_db_korquad_full_context_rag/ # (Gitignore) ChromaDB 벡터 데이터베이스 파일
├── db_setup.py                    # (독립 실행) 벡터 DB 초기 구축 스크립트
├── .env                           # 환경 변수 설정 파일 (모델 ID, DB 경로 등)
├── .gitignore                     # Git 추적 제외 목록
├── requirements.txt               # 프로젝트 의존성 라이브러리 목록
├── README.md                      # 프로젝트 개요 및 실행 가이드
├── run.sh                         # (편의) 초기 설정부터 서버 실행까지 자동화하는 셸 스크립트
└── STRUCTURE.md                   # 프로젝트 구조 설명 파일