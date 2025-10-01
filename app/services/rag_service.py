import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from app.core.config import settings
from app.models.schemas import SourceDocument

# --- 모델 및 벡터DB 로드 (싱글톤 패턴) ---
_llm = None
_embeddings = None
_vectordb = None
_retriever = None

def _load_quantized_model(model_id: str, device: str) -> HuggingFacePipeline:
    """4비트 양자화된 LLM을 로드합니다."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, repetition_penalty=1.1)
    return HuggingFacePipeline(pipeline=pipe)

def get_llm():
    """LLM을 로드하거나 이미 로드된 경우 반환합니다."""
    global _llm
    if _llm is None:
        print(f"LLM '{settings.LLM_MODEL_ID}' 로드 중...")
        _llm = _load_quantized_model(settings.LLM_MODEL_ID, settings.DEVICE_TYPE)
    return _llm

def get_embeddings():
    """임베딩 모델을 로드하거나 이미 로드된 경우 반환합니다."""
    global _embeddings
    if _embeddings is None:
        print(f"임베딩 모델 '{settings.EMBED_MODEL_ID}' 로드 중...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBED_MODEL_ID,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

def get_vectordb():
    """벡터DB를 로드하거나 이미 로드된 경우 반환합니다."""
    global _vectordb
    if not os.path.exists(settings.CHROMA_DB_PATH):
        raise FileNotFoundError(f"'{settings.CHROMA_DB_PATH}' 에서 벡터 DB를 찾을 수 없습니다. 'db_setup.py'를 먼저 실행하세요.")
    if _vectordb is None:
        print(f"벡터DB '{settings.CHROMA_DB_PATH}' 로드 중...")
        _vectordb = Chroma(persist_directory=settings.CHROMA_DB_PATH, embedding_function=get_embeddings())
    return _vectordb

def get_retriever(k: int = 5):
    """Retriever를 생성합니다."""
    return get_vectordb().as_retriever(search_kwargs={"k": k})

# --- Few-Shot RAG 서비스 로직 ---

def _format_docs(docs):
    """검색된 문서를 프롬프트에 맞게 포맷팅합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

def few_shot_rag_invoke(question: str, k_fewshot: int):
    """
    Few-Shot RAG 체인을 실행하여 질문에 대한 답변을 생성합니다.

    Args:
        question (str): 사용자 질문
        k_fewshot (int): Few-Shot 예시로 사용할 문서의 수

    Returns:
        dict: 답변, 출처 문서, 사용된 Few-Shot 예시 수를 포함하는 딕셔너리
    """
    llm = get_llm()
    retriever = get_retriever(k=k_fewshot + 1) # 답변 근거용 문서 1개 + few-shot 예시 k개

    # 1. 관련 문서 검색
    retrieved_docs = retriever.invoke(question)
    
    # 2. Few-Shot 예시와 실제 컨텍스트 분리
    if len(retrieved_docs) > 1 and k_fewshot > 0:
        context_doc = retrieved_docs[0] # 가장 유사도 높은 문서를 답변 근거로 사용
        few_shot_examples = retrieved_docs[1:k_fewshot + 1]
        num_examples_used = len(few_shot_examples)
    else: # 검색된 문서가 부족할 경우
        context_doc = retrieved_docs[0] if retrieved_docs else None
        few_shot_examples = []
        num_examples_used = 0

    # 3. 프롬프트 템플릿 정의
    template = """당신은 주어진 예시와 정보를 바탕으로 질문에 답변하는 AI 어시스턴트입니다. 항상 친절하고 상세하게 답변해주세요.
        만약 주어진 정보에 답변의 근거가 없다면, "정보가 부족하여 답변할 수 없습니다."라고 솔직하게 말해주세요.

        [Few-Shot 예시]
        {few_shot_examples}

        [정보]
        {context}

        [질문]
        {question}

        [답변]
        """
    prompt = PromptTemplate.from_template(template)

    # 4. RAG 체인 구성
    rag_chain = (
        {
            "few_shot_examples": lambda x: _format_docs(few_shot_examples),
            "context": lambda x: context_doc.page_content if context_doc else "제공된 정보 없음",
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 체인 실행
    answer = rag_chain.invoke(question)

    # 6. 출처 문서 포맷팅
    source_documents = []
    if context_doc:
        source_documents.append(SourceDocument(
            title=context_doc.metadata.get('title', 'N/A'),
            retrieved_question=context_doc.metadata.get('question', 'N/A'),
            content_snippet=context_doc.page_content,
            is_fewshot=False
        ))
    for doc in few_shot_examples:
        source_documents.append(SourceDocument(
            title=doc.metadata.get('title', 'N/A'),
            retrieved_question=doc.metadata.get('question', 'N/A'),
            content_snippet=doc.page_content,
            is_fewshot=True
        ))

    return {
        "answer": answer,
        "source_documents": source_documents,
        "few_shot_examples_used": num_examples_used
    }
