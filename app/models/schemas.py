from typing import List, Optional
from pydantic import BaseModel, Field

class RAGQueryRequest(BaseModel):
    """
    RAG 질의응답 API의 입력 형식
    """
    question: str = Field(..., description="사용자의 질문", example="유엔이 설립된 연도는?")
    k_fewshot: Optional[int] = Field(3, description="Few-Shot 예시로 사용할 검색 결과의 수")

class SourceDocument(BaseModel):
    """
    답변의 근거가 되는 출처 문서의 형식
    """
    title: str = Field(description="문서의 제목", example="유엔")
    retrieved_question: str = Field(description="검색된 원본 질문", example="유엔은 언제 설립되었는가?")
    content_snippet: str = Field(description="검색된 내용의 일부", example="유엔(UN)은 국제 연합(United Nations)의 약자로, 1945년 10월 24일에 설립된...")
    is_fewshot: bool = Field(description="이 문서가 Few-Shot 예시로 사용되었는지 여부", example=True)

class RAGAnswerResponse(BaseModel):
    """
    RAG 질의응답 API의 출력 형식
    """
    answer: str = Field(description="생성된 답변", example="1945년")
    source_documents: List[SourceDocument] = Field(description="답변의 근거가 되는 출처 문서 목록")
    few_shot_examples_used: int = Field(description="실제로 Few-Shot 프롬프트에 포함된 예시의 수", example=3)
