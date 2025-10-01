import os
import sys
import torch
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# src 디렉터리를 시스템 경로에 추가
sys.path.append(os.path.abspath('src'))

from dataload import load_and_process_data, create_vector_db, CHROMA_DB_PATH, EMBED_MODEL_ID
from llm_model import load_quantized_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    """
    데이터 기반 추론을 위한 메인 실행 함수
    """
    # 1. 벡터 DB 준비
    # ChromaDB가 디스크에 없으면 데이터 로드 및 벡터 DB 생성
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"'{CHROMA_DB_PATH}'가 존재하지 않아 새로 생성합니다.")
        documents = load_and_process_data()
        if documents:
            create_vector_db(documents)
        else:
            print("처리할 문서가 없습니다. 프로그램을 종료합니다.")
            return
    else:
        print(f"'{CHROMA_DB_PATH}'가 존재합니다.")

    # 2. 임베딩 모델 및 LLM 로드
    print("\n모델을 로드합니다...")
    try:
        # 임베딩 모델 로드
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # LLM 모델 로드
        llm_model_id = "kakaocorp/kanana-1.5-8b-instruct-2505"
        llm = load_quantized_model(llm_model_id)
        print("임베딩 모델과 LLM을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        return

    # 3. 벡터 DB 로드 및 Retriever 설정
    print("벡터 DB를 로드하고 Retriever를 설정합니다.")
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 4. RAG(Retrieval-Augmented Generation) 체인 구성
    print("RAG 체인을 구성합니다.")
    
    # 프롬프트 템플릿 정의
    template = """
    당신은 주어진 정보를 바탕으로 질문에 답변하는 AI 어시스턴트입니다. 
    항상 친절하고 상세하게 답변해주세요. 만약 정보가 부족하여 답변할 수 없다면, "정보가 부족하여 답변할 수 없습니다."라고 솔직하게 말해주세요.

    [정보]
    {context}

    [질문]
    {question}

    [답변]
    """
    prompt = PromptTemplate.from_template(template)

    # RAG 체인 정의
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 추론 실행
    print("\n추론을 시작합니다...")
    question = "임진왜란이 발발한 연도는?"

    try:
        print(f"질문: {question}")
        answer = rag_chain.invoke(question)
        
        print("\n[최종 답변]")
        print(answer)

    except Exception as e:
        print(f"추론 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
