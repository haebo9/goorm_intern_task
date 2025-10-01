import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.models.schemas import SourceDocument

# --- 모델 및 벡터DB 로드 (싱글톤 패턴) ---
_tokenizer = None
_model = None
_embeddings = None
_vectordb = None

def get_tokenizer_and_model():
    """Tokenizer와 4비트 양자화된 LLM을 로드합니다."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"LLM '{settings.LLM_MODEL_ID}' 로드 중...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        _tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map=settings.DEVICE_TYPE,
            trust_remote_code=True,
        )
        _model.eval()
    return _tokenizer, _model

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

def get_retriever(k: int = settings.DEFAULT_K_FEWSHOT):
    """Retriever를 생성하거나 반환합니다."""
    vectordb = get_vectordb()
    return vectordb.as_retriever(search_kwargs={'k': k})

def initialize_rag_system():
    """
    서버 시작 시 RAG 시스템에 필요한 모든 구성 요소를 미리 로드합니다.
    """
    print("RAG 시스템 초기화 시작: 임베딩, 벡터DB, LLM 로드")
    get_embeddings()
    get_vectordb()
    get_tokenizer_and_model()
    print("RAG 시스템 초기화 완료.")

# --- Few-Shot RAG 서비스 로직 ---

def _extract_answer_snippet(context: str, answer: str) -> str:
    """정답 텍스트 시작 위치부터 첫 마침표까지의 스니펫을 추출합니다."""
    if not answer or answer not in context:
        return context[:200] + "..."

    try:
        start_index = context.find(answer)
        if start_index == -1:
            return context[:200] + "..."

        end_index = context.find('.', start_index)
        
        if end_index != -1:
            snippet = context[start_index : end_index + 1]
        else:
            snippet = context[start_index : start_index + 200] + "..."
        
        return snippet.strip()
    except Exception:
        return context[:200] + "... (스니펫 추출 오류)"

def few_shot_rag_invoke(question: str, k_fewshot: int):
    """
    Few-Shot RAG 체인을 실행하여 질문에 대한 답변을 생성합니다.
    """
    tokenizer, model = get_tokenizer_and_model()
    retriever = get_retriever(k=k_fewshot + 1)

    retrieved_docs = retriever.invoke(question)

    # 검색된 문서가 없으면 바로 반환
    if not retrieved_docs:
        return {
            "answer": "관련 정보를 찾을 수 없습니다.",
            "source_documents": [],
            "few_shot_examples_used": 0
        }

    context_doc = retrieved_docs[0]
    few_shot_examples_docs = retrieved_docs[1:k_fewshot + 1] if len(retrieved_docs) > 1 and k_fewshot > 0 else []
    num_examples_used = len(few_shot_examples_docs)

    few_shot_examples_text = "".join([
        f"[예시 질문]: {doc.metadata['question']}\n[예시 출처]: {doc.page_content}\n[예시 답변]: {doc.metadata['answer']}\n"
        for doc in few_shot_examples_docs
    ])

    template = """## 임무:

        제공된 '최종 출처' 정보만을 사용하여 '최종 질문'에 가장 정확하게 답변하십시오. 출처에 답이 없으면 '주어진 정보로는 답을 찾을 수 없습니다.'라고 답하세요. 답변은 반드시 출처에 명시된 용어로 작성하십시오.

        ## 학습 예시:
        {few_shot_examples}

        ## 최종 출처:
        {context}

        ## 최종 질문:
        {question}

        [최종 답변]:
        """
    
    final_prompt_content = template.format(
        few_shot_examples=few_shot_examples_text,
        context=context_doc.page_content,
        question=question
    )

    messages = [{"role": "user", "content": final_prompt_content}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
        )
    
    response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    snippet = _extract_answer_snippet(context_doc.page_content, response_text)
    source_documents = [SourceDocument(
        title=context_doc.metadata.get('title', 'N/A'),
        retrieved_question=context_doc.metadata.get('question', 'N/A'),
        content_snippet=snippet,
        is_fewshot=False
    )]

    return {
        "answer": response_text,
        "source_documents": source_documents,
        "few_shot_examples_used": num_examples_used
    }