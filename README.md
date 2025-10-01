# KorQuAD 1.0 기반 RAG LLM REST API 서비스

이 프로젝트는 KorQuAD 1.0 데이터셋을 기반으로 한 RAG(Retrieval-Augmented Generation) LLM 질의응답 API 서비스를 구축하는 것을 목표로 합니다. Vector DB로는 ChromaDB를 사용합니다.

## 개발 환경 설정

1.  **가상 환경 생성 및 활성화:**

    프로젝트의 의존성을 독립적으로 관리하기 위해 가상 환경을 생성하고 활성화합니다.

    ```bash
    # 가상 환경 생성
    python3 -m venv .venv

    # 가상 환경 활성화 (macOS/Linux)
    source .venv/bin/activate
    ```
    *(참고: Windows 환경에서는 `source` 대신 `\.venv\Scripts\activate` 명령어를 사용합니다.)*

2.  **필요 라이브러리 설치:**

    `requirements.txt` 파일에 명시된 라이브러리들을 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```