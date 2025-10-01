import os
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd

# --- 설정 ---
# README에 명시된 고성능 한국어 임베딩 모델 사용
EMBED_MODEL_ID = "dragonkue/bge-m3-ko" 
# STRUCTURE.md에 명시된 DB 경로 사용
CHROMA_DB_PATH = "data/chroma_db_korquad_full_context_rag" 
DATASET_ID = "squad_kor_v1"

def load_and_process_data():
    """
    KorQuAD 1.0 데이터셋을 불러와서 질문, 답변, 컨텍스트를 포함하는 Document 객체 리스트를 생성합니다.
    """
    print(f"'{DATASET_ID}' 데이터셋 로드 중...")
    dataset = load_dataset(DATASET_ID, split='train')
    
    documents = []
    # 중복 컨텍스트를 방지하기 위해 이미 처리된 context_id를 기록
    processed_contexts = set()

    for item in dataset:
        context = item['context']
        # context가 중복되지 않은 경우에만 문서 추가
        if context not in processed_contexts:
            # Document 객체 생성
            # page_content: 전체 컨텍스트를 저장하여 RAG가 풍부한 정보에 접근하도록 함
            # metadata: 검색 및 필터링에 사용할 수 있는 정보 저장
            doc = Document(
                page_content=context,
                metadata={
                    'title': item['title'],
                    'question': item['question'], # 이 컨텍스트와 가장 관련 높은 대표 질문
                    'answer': item['answers']['text'][0] # 대표 질문에 대한 답변
                }
            )
            documents.append(doc)
            processed_contexts.add(context)

    print(f"총 {len(documents)}개의 고유한 컨텍스트 문서를 생성했습니다.")
    return documents

def create_vector_db(documents):
    """
    Document 리스트를 임베딩하여 ChromaDB에 저장합니다.
    """
    if not documents:
        print("저장할 문서가 없습니다.")
        return None

    print(f"임베딩 모델 '{EMBED_MODEL_ID}' 로드 중...")
    # README에 명시된 대로, 한국어에 특화된 고성능 임베딩 모델 사용
    # GPU를 사용하도록 설정
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("텍스트를 청크 단위로 분할하는 중...")
    # 컨텍스트가 길 수 있으므로 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    print(f"ChromaDB에 벡터 저장 중... (경로: '{CHROMA_DB_PATH}')")
    # 분할된 문서를 기반으로 ChromaDB 생성 및 영구 저장
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    print(f"'{CHROMA_DB_PATH}' 경로에 벡터 DB를 성공적으로 저장했습니다.")
    return vectordb

if __name__ == "__main__":
    print("--- 벡터 데이터베이스 구축 스크립트 시작 ---")
    
    # 데이터 디렉토리 생성
    os.makedirs(os.path.dirname(CHROMA_DB_PATH), exist_ok=True)

    # 1. 데이터 로드 및 전처리
    docs = load_and_process_data()
    
    # 2. 벡터 DB 생성 및 저장
    create_vector_db(docs)

    print("--- 모든 작업 완료 ---")
