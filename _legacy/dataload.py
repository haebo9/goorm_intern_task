import os
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 임베딩 모델 설정
EMBED_MODEL_ID = "jhgan/ko-sbert-nli"
# ChromaDB 저장 경로
CHROMA_DB_PATH = "chroma_db"

def load_and_process_data():
    """KorQuAD 1.0 데이터셋을 불러와서 질문과 답변 쌍으로 문서를 생성합니다."""
    print("KorQuAD 1.0 데이터셋을 불러오는 중...")
    dataset = load_dataset("squad_kor_v1")['train']
    
    documents = []
    for data in dataset:
        # 질문과 답변을 하나의 텍스트로 결합하여 Document 객체 생성
        text = f"질문: {data['question']}\n답변: {data['answers']['text'][0]}"
        metadata = {
            'title': data['title'],
            'context': data['context'],
            'question': data['question']
        }
        documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"총 {len(documents)}개의 문서를 생성했습니다.")
    return documents

def create_vector_db(documents):
    """문서들을 임베딩하여 ChromaDB에 저장합니다."""
    print("임베딩 모델을 불러오는 중...")
    # HuggingFace 임베딩 모델 초기화
    # CUDA가 아닌 CPU를 사용하도록 model_kwargs 수정
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("텍스트를 청크 단위로 분할하는 중...")
    # 텍스트를 적절한 크기로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    print("ChromaDB에 벡터를 저장하는 중...")
    # 분할된 텍스트를 기반으로 ChromaDB 벡터 스토어 생성 및 저장
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    print(f"'{CHROMA_DB_PATH}' 경로에 벡터 DB를 성공적으로 저장했습니다.")
    return vectordb

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    docs = load_and_process_data()
    
    # 벡터 DB 생성 및 저장
    if docs:
        create_vector_db(docs)
    else:
        print("처리할 문서가 없습니다.")