# 프로젝트 구조: GOORM_INTERN_TASK

프로젝트는 FastAPI 모범 사례를 따라 핵심 로직, API 엔드포인트, 설정을 명확하게 분리하여 관리하고 있습니다.

```bash
GOORM_INTERN_TASK/
├── .venv/                         # Python 가상 환경 (Git .gitignore에 추가)
├── app/                           # (핵심) FastAPI 애플리케이션 모듈
│   ├── api/                       #    API 라우터 정의
│   │   └── endpoints.py           #        -> RAG 질의응답 엔드포인트 정의 및 서비스 호출
│   ├── core/                      #    핵심 설정 및 상수
│   │   └── config.py              #        -> LLM 이름, DB 경로, K값, 환경 변수 로드
│   ├── services/                  #    비즈니스 로직 (Controller/Service 계층)
│   │   └── rag_service.py         #        -> Few-Shot RAG 추론 로직 (few_shot_rag_invoke) 구현
│   ├── models/                    #    Pydantic 데이터 모델
│   │   └── schemas.py             #        -> API 입출력 형식 정의 (RAGQueryRequest, RAGAnswerResponse)
│   └── main.py                    #    FastAPI 앱 인스턴스 생성 및 라우터 연결 (앱의 진입점)
├── db_setup.py                    # (독립 실행) 벡터 DB 초기 구축 스크립트
├── data/                          # DB 및 캐시 파일 저장소
│   └── chroma_db_korquad_full_context_rag/ # Chroma DB 파일
├── .env                           # 환경 변수 설정 파일 (FastAPI 설정 및 기타 인증 정보)
├── requirements.txt               # 프로젝트 의존성 라이브러리 목록
└── README.md                      # 프로젝트 개요 및 실행 가이드