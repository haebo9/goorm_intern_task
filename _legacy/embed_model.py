from langchain_community.embeddings import HuggingFaceEmbeddings

# 임베딩 모델 ID
EMBED_MODEL_ID = "jhgan/ko-sbert-nli"

def get_embedding_model(device='cpu'):
    """
    HuggingFace 임베딩 모델을 불러옵니다.
    
    Args:
        device (str): 모델을 로드할 장치 ('cpu' 또는 'cuda')
    
    Returns:
        HuggingFaceEmbeddings: 로드된 임베딩 모델
    """
    print(f"임베딩 모델 '{EMBED_MODEL_ID}'을(를) '{device}'에 로드합니다.")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

if __name__ == '__main__':
    # 모듈 단독 실행 시 테스트
    embedding_model = get_embedding_model()
    print("임베딩 모델 로드 완료!")
    
    test_text = "이것은 임베딩 모델 테스트입니다."
    vector = embedding_model.embed_query(test_text)
    print(f"'{test_text}'의 임베딩 벡터 (일부):")
    print(vector[:5])