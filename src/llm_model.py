import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def load_quantized_model(model_id: str) -> HuggingFacePipeline:
    """
    지정된 모델 ID를 4비트 양자화하여 불러옵니다.

    Args:
        model_id (str): Hugging Face Hub의 모델 ID

    Returns:
        HuggingFacePipeline: LangChain에서 사용할 수 있는 양자화된 모델 파이프라인
    """
    # 4비트 양자화를 위한 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 양자화 설정을 적용하여 모델 불러오기
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",  # 사용 가능한 GPU에 모델 레이어를 자동으로 분배
        trust_remote_code=True, # 모델 저장소의 코드를 신뢰하고 실행
    )

    # Transformers 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        repetition_penalty=1.1  # 반복을 줄이기 위한 패널티
    )

    # LangChain의 HuggingFacePipeline으로 래핑
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

if __name__ == '__main__':
    # 사용할 모델 ID (카카오의 Kanana 모델)
    MODEL_ID = "kakaocorp/kanana-1.5-8b-instruct-2505"

    print(f"'{MODEL_ID}' 모델을 4비트 양자화하여 불러오는 중...")
    
    try:
        llm_pipeline = load_quantized_model(MODEL_ID)
        print("모델 불러오기 완료!")

        # 테스트 프롬프트
        prompt = "대한민국의 수도는 어디인가요?"
        print(f"\n질문: {prompt}")
        
        # 모델 실행
        result = llm_pipeline.invoke(prompt)
        print(f"답변: {result}")

    except ImportError as e:
        print(f"오류: {e}")
        print("모델을 불러오기 위해 필요한 라이브러리가 설치되지 않았을 수 있습니다.")
        print("다음 명령어를 실행하여 'bitsandbytes'와 'accelerate'를 설치해주세요:")
        print("pip install bitsandbytes accelerate")
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
