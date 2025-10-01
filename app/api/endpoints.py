from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQueryRequest, RAGAnswerResponse
from app.services import rag_service

router = APIRouter()

@router.post("/answer", response_model=RAGAnswerResponse)
def get_rag_answer(request: RAGQueryRequest):
    """
    Few-Shot RAG 모델을 사용하여 사용자의 질문에 답변합니다.

    - **question**: 사용자의 질문 텍스트
    - **k_fewshot**: 답변 생성 시 참고할 Few-Shot 예시의 수 (기본값: 3)
    """
    try:
        result = rag_service.few_shot_rag_invoke(
            question=request.question,
            k_fewshot=request.k_fewshot
        )
        return RAGAnswerResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"서비스 준비 중: {e}")
    except Exception as e:
        # 프로덕션 환경에서는 로깅을 통해 에러를 기록하는 것이 좋습니다.
        print(f"Error during RAG invocation: {e}")
        raise HTTPException(status_code=500, detail="답변 생성 중 오류가 발생했습니다.")
