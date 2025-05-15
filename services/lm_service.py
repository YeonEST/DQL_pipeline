# services/lm_service.py
"""
LM Studio 연동 서비스

LM Studio API를 사용하여 텍스트 생성 및 마크다운 변환을 처리하는 서비스입니다.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LMStudioService:
    """LM Studio 연동 서비스"""

    def __init__(
            self,
            api_url: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            api_key: str = os.getenv("LM_STUDIO_KEY", "lm-studio-local")
    ):
        """
        LMStudioService 초기화

        Args:
            api_url: LM Studio API URL
            api_key: API 키
        """
        self.api_url = api_url
        self.api_key = api_key
        self.models_endpoint = f"{api_url}/models"
        self.chat_endpoint = f"{api_url}/chat/completions"
        self.completions_endpoint = f"{api_url}/completions"

    def test_connection(self) -> bool:
        """
        LM Studio 연결 테스트

        Returns:
            연결 성공 여부
        """
        try:
            models = self.get_available_models()
            if "data" in models and len(models["data"]) > 0:
                logger.info(f"LM Studio 연결 성공: {len(models['data'])}개 모델 발견")
                return True
            else:
                logger.warning("LM Studio 연결됨, 사용 가능한 모델 없음")
                return False
        except Exception as e:
            logger.error(f"LM Studio 연결 테스트 실패: {e}", exc_info=True)
            return False

    # LMStudioService 클래스의 get_available_models 메서드
    def get_available_models(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        사용 가능한 모델 목록 가져오기

        Args:
            use_cache: 캐시 사용 여부

        Returns:
            사용 가능한 모델 목록과 정보
        """
        from utils.presets import load_api_cache, save_api_cache

        # 캐시 사용인 경우 캐시 확인
        if use_cache:
            cached_data = load_api_cache("lm_studio_models", max_age_hours=1)
            if cached_data:
                logger.info("캐시된 LM Studio 모델 목록 사용")
                return cached_data

        try:
            logger.info(f"사용 가능한 모델 목록 요청: {self.models_endpoint}")
            response = requests.get(
                self.models_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5  # 타임아웃 추가
            )
            response.raise_for_status()
            models_data = response.json()

            # 데이터 캐싱 (1시간 유효)
            if use_cache:
                save_api_cache("lm_studio_models", models_data, max_age_hours=1)

            return models_data
        except Exception as e:
            logger.error(f"모델 목록 가져오기 실패: {e}", exc_info=True)
            return {"data": []}

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        특정 모델에 대한 상세 정보 가져오기

        Args:
            model_id: 모델 ID

        Returns:
            모델 정보 (컨텍스트 길이, 최대 토큰 수 등)
        """
        try:
            logger.info(f"모델 정보 요청: {model_id}")

            # 1. 모델 목록에서 기본 정보 확인
            models_data = self.get_available_models()
            model_info = {
                "id": model_id,
                "context_length": 32000,  # 기본값을 더 큰 값으로 설정
                "context_length_source": "default"
            }

            if "data" in models_data:
                for model in models_data["data"]:
                    if model["id"] == model_id:
                        model_info.update(model)
                        break

            # 2. 모델 세부 정보 요청 (모델 ID 자체에 요청 보내기)
            try:
                model_endpoint = f"{self.models_endpoint}/{model_id}"
                response = requests.get(
                    model_endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5  # 타임아웃 설정
                )
                if response.status_code == 200:
                    detailed_info = response.json()
                    model_info.update(detailed_info)
                    if "context_length" in detailed_info:
                        model_info["context_length_source"] = "api"
            except Exception as e:
                logger.warning(f"모델 상세 정보 요청 실패: {e}")
                # 실패해도 계속 진행

            # 3. 모델 카드에서 일반적인 정보 추정
            # 모델명에서 추정 (일부 모델은 이름에 context 크기가 포함됨)
            if model_info.get("context_length_source") == "default":
                model_name = model_id.lower()

                # 많이 사용되는 모델들의 컨텍스트 크기 맵 (업데이트됨)
                known_models = {
                    "llama-2-7b": 4096,
                    "llama-2-13b": 4096,
                    "llama-2-70b": 4096,
                    "llama-3-8b": 8192,
                    "llama-3-70b": 8192,
                    "mistral-7b": 8192,
                    "mixtral-8x7b": 32768,
                    "mixtral-8x22b": 65536,
                    "gemma-7b": 8192,
                    "gemma-2b": 8192,
                    "gemma-2-9b": 8192,
                    "gemma-2-27b": 8192,
                    "phi-2": 2048,
                    "phi-3": 4096,
                    "phi-3-mini": 8192,
                    "phi-3-medium": 8192,
                    "phi-3-vision": 8192,
                    "codellama-7b": 16384,
                    "codellama-13b": 16384,
                    "codellama-34b": 16384,
                    "openhermes": 8192,
                    "openchat": 8192,
                    "falcon": 2048,
                    "falcon-40b": 2048,
                    "falcon-180b": 2048,
                    "mpt-7b": 2048,
                    "mpt-30b": 8192,
                    "claude": 100000,
                    "qwen": 8192,
                    "qwen-7b": 8192,
                    "qwen-14b": 8192,
                    "qwen-72b": 32768,
                    "nous-hermes": 8192,
                    "vicuna": 4096,
                    "vicuna-7b": 4096,
                    "vicuna-13b": 4096,
                    "wizardlm": 8192,
                    "mistral-small": 32768,
                    "mistral-medium": 32768
                }

                # 모델명에서 컨텍스트 크기 찾기
                context_size = None
                for known_model, size in known_models.items():
                    if known_model in model_name:
                        context_size = size
                        logger.info(f"알려진 모델 '{known_model}'을 기반으로 컨텍스트 크기 추정: {size}")
                        break

                # 명시적인 컨텍스트 크기 정보가 있는지 확인 (이름에 포함된 경우)
                if not context_size:
                    import re
                    # 8k, 32k, 100k 등의 패턴 찾기
                    context_patterns = re.findall(r'[-_\s](\d+)[kK][-_\s]?', model_name)
                    if context_patterns:
                        try:
                            # k를 1000으로 변환
                            context_size = int(context_patterns[0]) * 1024
                            logger.info(f"모델명에서 컨텍스트 크기 추출: {context_size}")
                        except:
                            pass

                # 컨텍스트 크기 정보 추가
                if context_size:
                    model_info["context_length"] = context_size
                    model_info["context_length_source"] = "estimated"

            # 4. completions API로 테스트 요청 보내 정보 수집
            if model_info.get("context_length_source") == "default":
                try:
                    # /v1/completions 엔드포인트로 테스트
                    test_response = requests.post(
                        self.completions_endpoint,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        json={
                            "model": model_id,
                            "prompt": "Hello",
                            "max_tokens": 10
                        },
                        timeout=300  # 타임아웃 설정
                    )
                    if test_response.status_code == 200:
                        response_data = test_response.json()
                        if "model" in response_data:
                            model_info["model"] = response_data["model"]
                        if "usage" in response_data:
                            model_info["usage_sample"] = response_data["usage"]
                except Exception as e:
                    logger.warning(f"모델 테스트 요청 실패: {e}")

            return model_info

        except Exception as e:
            logger.error(f"모델 정보 가져오기 실패: {e}", exc_info=True)
            return {
                "id": model_id,
                "context_length": 32000,  # 더 큰 기본값으로 변경
                "context_length_source": "fallback",
                "error": str(e)
            }

    def markdown_conversion(
            self,
            text: str,
            model: str,
            prompt_template: str = "",
            temperature: float = 0.2,
            max_tokens: int = 4096,
            top_p: float = 0.95,
            top_k: int = 40,
            min_p: float = 0.05,
            context_template: str = "custom"
    ) -> Dict[str, Any]:
        """
        텍스트를 마크다운으로 변환

        Args:
            text: 변환할 텍스트
            model: 사용할 모델명
            prompt_template: 프롬프트 템플릿 (비어있으면 기본 템플릿 사용)
            temperature: 온도 (창의성 조절, 낮을수록 일관성 있는 결과)
            max_tokens: 최대 토큰 수
            top_p: 확률 질량의 상위 p% 내에서 토큰 선택 (nucleus sampling)
            top_k: 확률 상위 k개 토큰 내에서 선택
            min_p: 확률 하한값 (min_p 미만인 토큰 제외)
            context_template: 컨텍스트 템플릿 타입 (custom, chatML, Alpaca, mistral)

        Returns:
            변환 결과
        """
        try:
            # 프롬프트 템플릿이 비어있으면 기본 템플릿 사용
            if not prompt_template.strip():
                prompt_template = """다음 텍스트를 깔끔한 마크다운 형식으로 변환해주세요. 불필요한 설명 없이 내용만 마크다운으로 정리해서 출력해주세요:

{text}

변환된 마크다운:"""

            # 프롬프트에 텍스트 삽입
            prompt = prompt_template.format(text=text)

            logger.info(f"마크다운 변환 요청: 모델 '{model}', 텍스트 길이: {len(text)}")

            # 컨텍스트 템플릿에 따른 시스템 메시지 설정
            system_message = "당신은 텍스트를 깔끔한 마크다운 형식으로 변환하는 전문가입니다. 필요 없는 부분은 제거하고 헤더, 목록, 표 등의 마크다운 요소를 적절히 사용하여 구조화된 문서를 만듭니다."

            # 컨텍스트 템플릿 선택
            messages = []
            if context_template == "chatML":
                # ChatML 포맷
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            elif context_template == "Alpaca":
                # Alpaca 포맷 (시스템 메시지를 유저 메시지에 포함)
                messages = [
                    {"role": "user", "content": f"### 지시사항:\n{system_message}\n\n### 입력:\n{prompt}\n\n### 응답:"}
                ]
            elif context_template == "mistral":
                # Mistral 포맷
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""}
                ]
            else:  # custom 또는 기타
                # 기본 ChatML 형식
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]

            # API 요청 데이터 준비
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p
            }

            # API 요청
            response = requests.post(
                self.chat_endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=request_data,
                timeout=60000  # 변환은 오래 걸릴 수 있으므로 더 긴 타임아웃 설정
            )
            response.raise_for_status()
            result = response.json()

            # 응답에서 마크다운 텍스트 추출
            if "choices" in result and len(result["choices"]) > 0:
                markdown_text = result["choices"][0]["message"]["content"]
                logger.info(f"마크다운 변환 성공: 응답 길이 {len(markdown_text)}")

                return {
                    "success": True,
                    "markdown": markdown_text,
                    "model": model,
                    "usage": result.get("usage", {})
                }
            else:
                logger.error(f"마크다운 변환 실패: 응답에 choices가 없습니다. 응답: {result}")
                return {
                    "success": False,
                    "error": "응답 포맷 오류",
                    "details": result
                }

        except Exception as e:
            logger.error(f"마크다운 변환 중 오류 발생: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }