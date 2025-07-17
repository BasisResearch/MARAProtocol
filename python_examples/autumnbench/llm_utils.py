from abc import ABC, abstractmethod
import os
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv
import backoff
from openai import OpenAI, AzureOpenAI
# from anthropic import Anthropic
# from ollama import Client
# from langchain_ollama.llms import OllamaLLM
import json
import asyncio
import requests
# from google import genai
import base64
import re

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

load_dotenv()


class Message:

    def __init__(self,
                 role: str,
                 content: str,
                 base64_images: Optional[List[str]] = None,
                 media_types: Optional[List[str]] = None):
        self.role = role
        self.content = content
        self.base64_images = base64_images
        self.media_types = media_types


class ModelNotSupportedError(Exception):
    """Raised when a model doesn't support requested functionality"""
    pass


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.supports_vision = False

    @abstractmethod
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        """Generate response from the model"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        """Generate response from the model asynchronously"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        raise NotImplementedError("Subclasses must implement this method")


class GeminiProvider(AIProvider):

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.supports_vision = True  # Gemini 2.0 Flash supports vision

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        # Format the message content - Gemini uses a chat format
        prompt_parts = [system_prompt, content]

        # Add images if provided
        if images and self.supports_vision:
            image_parts = []
            for img_path in images:
                if img_path.startswith("http"):
                    # Handle URL images
                    image_parts.append({
                        "mime_type":
                        "image/jpeg",  # Assuming JPEG, adjust as needed
                        "data":
                        self._get_image_data_from_url(img_path)
                    })
                else:
                    # Assume base64-encoded image
                    image_parts.append({
                        "mime_type":
                        "image/jpeg",  # Assuming JPEG, adjust as needed
                        "data": img_path
                    })

            # Insert images after the content
            prompt_parts = [system_prompt, content, *image_parts]

        # Generate response
        response = self.client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt_parts)

        # Return the generated text
        return response.text

    def _get_image_data_from_url(self, url: str) -> bytes:
        """Download image data from a URL"""
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        """Asynchronous generation - currently just calls the sync version"""
        # For simplicity, we're using the synchronous method in an async wrapper
        # This can be optimized with a proper async implementation
        return self.generate(system_prompt, content, images)

    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        prompt_parts = [
            {
                "role": "system",
                "content": system_prompt
            },
        ]
        for part in prompt_parts:
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img}"
                            }
                        })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in part.base64_images)
            if part.role == "user":
                prompt_parts.append({"role": "user", "content": content})
            elif part.role == "assistant":
                prompt_parts.append({"role": "assistant", "content": content})
        return self.client.models.generate_content(model="gemini-2.0-flash",
                                                   contents=prompt_parts).text


class OpenAIProvider(AIProvider):

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.supports_vision = True

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content

    def _format_content(self,
                        content: str,
                        images: Optional[List[str]] = None) -> Any:
        if not images:
            return content

        return [{
            "type": "text",
            "text": content
        }, *[{
            "type": "image_url",
            "image_url": {
                "url": img
            }
        } for img in images]]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        prompt_parts = [Message("system", system_prompt), *prompt_parts]
        final_prompt_parts = []
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img}"
                            }
                        })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in part.base64_images)
            final_prompt_parts.append({"role": part.role, "content": content})
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=final_prompt_parts,
            max_tokens=8192,
        ).choices[0].message.content


class AzureOpenAIProvider(AIProvider):

    def __init__(self, model_name: str = "openai/gpt-4o"):
        super().__init__(model_name)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version="2024-02-15-preview",
        )
        self.supports_vision = False

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_from_history(
            self,
            messages: List[dict],
            generate_options: Optional[Dict[str, Any]] = {}) -> str:

        response = self.client.chat.completions.create(model=self.model_name,
                                                       messages=messages,
                                                       max_tokens=8192,
                                                       **generate_options)
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content, messages

    def _format_content(self,
                        content: str,
                        images: Optional[List[str]] = None) -> Any:
        if not images:
            return content

        return [{
            "type": "text",
            "text": content
        }, *[{
            "type": "image_url",
            "image_url": {
                "url": img
            }
        } for img in images]]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_formatted_prompt(
            self,
            system_prompt: str,
            prompt_parts: List[Message],
            generate_options: Optional[Dict[str, Any]] = {}) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        prompt_parts = [Message("system", system_prompt), *prompt_parts]
        final_prompt_parts = []
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img}"
                            }
                        })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in part.base64_images)
            final_prompt_parts.append({"role": part.role, "content": content})
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=final_prompt_parts,
            max_tokens=8192,
            **generate_options).choices[0].message.content, final_prompt_parts


class OpenRouterProvider(AIProvider):

    def __init__(self, model_name: str = "openai/gpt-4o"):
        super().__init__(model_name)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.getenv('OPENROUTER_API_KEY'))
        self.supports_vision = False

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_from_history(
            self,
            messages: List[dict],
            generate_options: Optional[Dict[str, Any]] = {}) -> str:

        response = self.client.chat.completions.create(model=self.model_name,
                                                       messages=messages,
                                                       max_tokens=8192,
                                                       **generate_options)
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content, messages

    def _format_content(self,
                        content: str,
                        images: Optional[List[str]] = None) -> Any:
        if not images:
            return content

        return [{
            "type": "text",
            "text": content
        }, *[{
            "type": "image_url",
            "image_url": {
                "url": img
            }
        } for img in images]]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_formatted_prompt(
            self,
            system_prompt: str,
            prompt_parts: List[Message],
            generate_options: Optional[Dict[str, Any]] = {}) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        prompt_parts = [Message("system", system_prompt), *prompt_parts]
        final_prompt_parts = []
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img}"
                            }
                        })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in part.base64_images)
            final_prompt_parts.append({"role": part.role, "content": content})
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=final_prompt_parts,
            max_tokens=8192,
            **generate_options).choices[0].message.content, final_prompt_parts


class ClaudeProvider(AIProvider):

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name)
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.supports_vision = True

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "assistant",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.content[0].text

    def _get_image_data_from_url(self, url: str) -> bytes:
        """Download image data from a URL"""
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def _format_content(self,
                        content: str,
                        images: Optional[List[str]] = None) -> Any:
        if not images:
            return content
        base64_images = []
        media_types = []
        for img in images:
            if img.startswith("http"):
                img_bytes = self._get_image_data_from_url(img)
                base64_images.append(
                    base64.b64encode(img_bytes).decode("utf-8"))
                if img.endswith(".png"):
                    media_type = "image/png"
                else:
                    media_type = "image/jpeg"
                media_types.append(media_type)
            else:
                base64_images.append(img)
                media_types.append("image/jpeg")
        return [{
            "type": "text",
            "text": content
        }, *[{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img
            }
        } for img, media_type in zip(base64_images, media_types)]]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        messages = [{
            "role": "assistant",
            "content": system_prompt
        }, {
            "role": "user",
            "content": self._format_content(content, images)
        }]

        response = await self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=8192,
        )
        return response.content[0].text

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        prompt_parts = [Message("assistant", system_prompt), *prompt_parts]
        final_prompt_parts = []
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img
                            }
                        })
                else:
                    for img in part.base64_images:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img
                            }
                        })
            final_prompt_parts.append({
                "role":
                "assistant" if part.role == "system" else part.role,
                "content":
                content
            })
        return self.client.messages.create(
            model=self.model_name,
            messages=final_prompt_parts,
            max_tokens=8192,
        ).content[0].text


class OllamaProvider(AIProvider):

    def __init__(self,
                 model_name: str = "llama3:8b",
                 base_url: Optional[str] = None):
        super().__init__(model_name)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self.client = Client(host=self.base_url)
        self.supports_vision = model_name in ["llama3.2-vision"]

        # For text-only models, we'll use LangChain's OllamaLLM
        if not self.supports_vision:
            self.llm = OllamaLLM(model=model_name, base_url=self.base_url)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        if self.supports_vision:
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": content,
                    "images": images or []
                }],
            )
            return response["message"]["content"]
        else:
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": content
                }],
            )
            return response["message"]["content"]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        if self.supports_vision:
            response = await self.client.chat(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": content,
                    "images": images or []
                }],
            )
            return response["message"]["content"]
        else:
            # Note: Using LangChain's async interface for text-only models
            response = await self.llm.agenerate([content])
            return response.generations[0][0].text

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        messages = [
            Message("assistant", system_prompt),
        ]
        final_prompt_parts = []
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                if part.media_types:
                    for img, media_type in zip(part.base64_images,
                                               part.media_types):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{img}"
                            }
                        })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    } for img in part.base64_images)
            final_prompt_parts.append({"role": part.role, "content": content})

        return self.client.chat(
            model=self.model_name,
            messages=final_prompt_parts,
        ).choices[0].message.content


class MLXClientProvider(AIProvider):
    """Client provider that connects to a local MLX server running on the host"""

    def __init__(
            self,
            model_name: str = "mlx-community/Phi-3-mini-4k-instruct-q4f16_1",
            base_url: str = None,
            tokenizer_config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLX Client Provider
        
        Args:
            model_name: Name of the model to use on the MLX server
            base_url: URL of the MLX server (default: http://host.docker.internal:8080)
            tokenizer_config: Optional tokenizer configuration
        """
        super().__init__(model_name)
        self.base_url = base_url or os.getenv(
            "MLX_SERVER_URL", "http://host.docker.internal:8080")
        self.tokenizer_config = tokenizer_config or {}
        self.supports_vision = False

    @backoff.on_exception(backoff.expo,
                          (requests.RequestException, json.JSONDecodeError),
                          max_tries=3)
    def generate(self,
                 system_prompt: str,
                 content: str,
                 images: Optional[List[str]] = None) -> str:
        """Generate using the MLX Server API"""
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        data = {
            "model_name": self.model_name,
            "system_prompt": system_prompt,
            "content": content,
            "tokenizer_config": self.tokenizer_config,
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.05
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
        except requests.RequestException as e:
            raise RuntimeError(f"MLX server request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid response from MLX server: {e}")

    @backoff.on_exception(backoff.expo,
                          (requests.RequestException, json.JSONDecodeError),
                          max_tries=3)
    async def agenerate(self,
                        system_prompt: str,
                        content: str,
                        images: Optional[List[str]] = None) -> str:
        """Generate asynchronously using the MLX Server API"""
        if images and not self.supports_vision:
            raise ModelNotSupportedError(
                f"Model {self.model_name} does not support vision tasks")

        import aiohttp

        data = {
            "model_name": self.model_name,
            "system_prompt": system_prompt,
            "content": content,
            "tokenizer_config": self.tokenizer_config,
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.05
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/generate",
                                        json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"MLX server request failed with status {response.status}: {error_text}"
                        )

                    result = await response.json()
                    return result["choices"][0]["text"]
        except aiohttp.ClientError as e:
            raise RuntimeError(f"MLX server request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid response from MLX server: {e}")

    def generate_formatted_prompt(self, system_prompt: str,
                                  prompt_parts: List[Message]) -> str:
        """Generate a formatted prompt from a list of prompt parts"""
        messages = [
            Message("assistant", system_prompt),
        ]
        for part in prompt_parts:
            assert isinstance(part,
                              Message), f"Prompt part {part} is not a Message"
            content = []
            if part.content:
                content.append({"type": "text", "text": part.content})
            if part.base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                } for img in part.base64_images)
            messages.append({"role": part.role, "content": content})
        return self.client.chat(
            model=self.model_name,
            messages=messages,
        ).text


class AIFactory:
    """Factory class to create AI provider instances"""

    @staticmethod
    def create_provider(provider_type: str,
                        model_name: str = None,
                        vision=False) -> AIProvider:
        if provider_type == "openai":
            return OpenAIProvider("gpt-4o")
        elif provider_type == "openrouter":
            return OpenRouterProvider(model_name)
        elif provider_type == "claude":
            return ClaudeProvider("claude-3-5-sonnet-20241022")
        elif provider_type == "ollama":
            OLLAMA_MODEL_TEXT = os.getenv("OLLAMA_MODEL_TEXT", "gemma2:9b")
            OLLAMA_MODEL_VISION = os.getenv("OLLAMA_MODEL_VISION",
                                            "llama3.2-vision")
            return OllamaProvider(
                OLLAMA_MODEL_TEXT if not vision else OLLAMA_MODEL_VISION)
        elif provider_type == "mlx":
            MLX_MODEL = os.getenv("MLX_MODEL",
                                  "mlx-community/Qwen2.5-14B-Instruct-bf16")
            MLX_SERVER_URL = os.getenv("MLX_SERVER_URL",
                                       "http://host.docker.internal:10080")

            # Parse tokenizer config
            tokenizer_config = {}
            mlx_tokenizer_config = os.getenv("MLX_TOKENIZER_CONFIG", "")
            if mlx_tokenizer_config:
                try:
                    tokenizer_config = json.loads(mlx_tokenizer_config)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse MLX_TOKENIZER_CONFIG as JSON: {mlx_tokenizer_config}"
                    )

            return MLXClientProvider(MLX_MODEL, MLX_SERVER_URL,
                                     tokenizer_config)
        elif provider_type == "llama-cpp":
            # Get model path from environment
            model_path = os.getenv("LLAMACPP_MODEL_PATH")
            n_ctx = int(os.getenv("LLAMACPP_N_CTX", "4096"))
            n_gpu_layers = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "-1"))
            return LlamaCppProvider(model_path, n_ctx, n_gpu_layers)
        elif provider_type == "gemini":
            return GeminiProvider()
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


async def citation_tool(provider_type: str,
                        answer: str,
                        references: list[str],
                        start_idx: int = 0):
    llm_provider = AIFactory.create_provider(provider_type)
    system_prompt = """
    You are a helpful biomedical expert that provides citations for the given answer with respect to the context.
    Given the answer, and the references with citation index, return the answer with citations.
    For example:
    Answer: The answer is 1.
    References: 
    [1] The first reference
    [2] The second reference
    [3] The third reference
    Output: The answer is 1 [1].

    The answer and references are given by the user below.
    """

    context = ""
    for reference in references:
        context += f"[{start_idx}] {reference}\n"
        start_idx += 1

    prompt = f"""
    Answer: {answer}
    References:
    {context}
    """

    return await llm_provider.agenerate(system_prompt, prompt), {
        i + start_idx: reference
        for i, reference in enumerate(references)
    }


def extract_json_response(response):
    """
    Extract the JSON response from the LLM output with improved robustness.
    """
    try:
        # Try different patterns for extracting JSON
        import json
        import re

        # First, check for markdown JSON code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response,
                               re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response,
                               re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Next, look for JSON without markdown formatting
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            json_str = json_match.group(1)
            # Try to find the first valid JSON object
            brace_count = 0
            start_idx = json_str.find('{')

            for i in range(start_idx, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # We found a complete JSON object
                        potential_json = json_str[start_idx:i + 1]
                        try:
                            return json.loads(potential_json)
                        except:
                            # Keep looking if this isn't valid JSON
                            continue

        # If we still don't have valid JSON, try to clean the response and parse again
        # Remove non-JSON characters that might appear before or after the JSON object
        clean_response = re.sub(r'^[^{]*', '', response)
        clean_response = re.sub(r'[^}]*$', '', clean_response)

        try:
            return json.loads(clean_response)
        except:
            pass

        # Last resort: look for key-value pairs and construct a dict
        pairs = re.findall(r'"([^"]+)":\s*([\d.]+)', response)
        if pairs:
            result = {}
            for key, value in pairs:
                try:
                    result[key] = float(value)
                except:
                    result[key] = value
            if result:
                return result

        return None
    except Exception as e:
        print(f"Error extracting JSON response: {e}")
        print(f"Raw response: {response}")
        return None


def extract_tagged_response(response):
    """
    Extract content from any XML-like tags in a response and return as a dictionary.
    Any text not included in tags is considered a "thought".
    
    Args:
        response (str): The response string with XML-like tags
        
    Returns:
        dict: A dictionary containing tag names as keys and their content as values,
              plus a 'thought' key containing any text outside of tags
    """
    result = {}

    # Find all XML-like tags and their content
    tag_pattern = r'<(\w+)>(.*?)</\1>'
    matches = re.findall(tag_pattern, response, re.DOTALL)

    for tag_name, content in matches:
        result[tag_name] = content.strip()

    # Extract all text outside of tags as "thought"
    # Remove all tagged content from the response
    response_without_tags = response
    for match in re.finditer(tag_pattern, response, re.DOTALL):
        response_without_tags = response_without_tags.replace(
            match.group(0), '', 1)

    # Clean up the remaining text and use it as thought
    thought_content = response_without_tags.strip()
    if thought_content:
        result['thought'] = thought_content

    return result


def test_openai():
    factory = AIFactory()
    openai_provider = factory.create_provider("openai")
    response = openai_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the capital of France?")
    print("OpenAI Response:", response)


def test_openai_vision():
    factory = AIFactory()
    openai_provider = factory.create_provider("openai")
    response = openai_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the content of this image?",
        images=[
            "https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/330px-Lenna_%28test_image%29.png"
        ])
    print("OpenAI Vision Response:", response)


def test_openai_messages():
    factory = AIFactory()
    openai_provider = factory.create_provider("openai")
    img_url = "https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/330px-Lenna_%28test_image%29.png"
    img_content = requests.get(img_url).content
    img_base64 = base64.b64encode(img_content).decode("utf-8")
    messages = [
        Message("user",
                "What is in this image?",
                base64_images=[img_base64],
                media_types=["image/png"])
    ]
    response = openai_provider.generate_formatted_prompt(
        system_prompt="You are a helpful assistant.", prompt_parts=messages)
    print("OpenAI Messages Response:", response)


def test_claude():
    factory = AIFactory()
    claude_provider = factory.create_provider("claude")
    response = claude_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the capital of France?")
    print("Claude Response:", response)


def test_claude_vision():
    factory = AIFactory()
    claude_provider = factory.create_provider("claude")
    response = claude_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the content of this image?",
        images=[
            "https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/330px-Lenna_%28test_image%29.png"
        ])
    print("Claude Vision Response:", response)


def test_claude_messages():
    factory = AIFactory()
    claude_provider = factory.create_provider("claude")
    img_url = "https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/330px-Lenna_%28test_image%29.png"
    img_content = requests.get(img_url).content
    img_base64 = base64.b64encode(img_content).decode("utf-8")
    messages = [
        Message("system", "You are a helpful assistant."),
        Message("user",
                "What is in this image?",
                base64_images=[img_base64],
                media_types=["image/png"])
    ]
    response = claude_provider.generate_formatted_prompt(
        system_prompt="You are a helpful assistant.", prompt_parts=messages)
    print("Claude Messages Response:", response)


def test_ollama():
    factory = AIFactory()
    ollama_provider = factory.create_provider("ollama")
    response = ollama_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the capital of France?")
    print("Ollama Response:", response)


def test_mlx():
    factory = AIFactory()
    mlx_provider = factory.create_provider("mlx")
    response = mlx_provider.generate(
        system_prompt="You are a helpful assistant.",
        content="What is the capital of France?")
    print("MLX Response:", response)


# Example usage
if __name__ == "__main__":
    # Example with text-only query
    factory = AIFactory()

    if os.getenv("OPENAI_API_KEY"):
        test_openai()
        test_openai_vision()
        test_openai_messages()

    if os.getenv("ANTHROPIC_API_KEY"):
        test_claude()
        test_claude_vision()
        test_claude_messages()
    try:
        test_ollama()
    except Exception as e:
        print(f"Ollama test error: {e}")
    # Test MLX if available
    try:
        test_mlx()
    except Exception as e:
        print(f"MLX test error: {e}")

    # Test llama-cpp if available
    if LLAMA_CPP_AVAILABLE and os.getenv("LLAMACPP_MODEL_PATH"):
        try:
            llama_cpp_provider = factory.create_provider("llama-cpp")
            response = llama_cpp_provider.generate(
                system_prompt="You are a helpful assistant.",
                content="What is the capital of France?")
            print("llama-cpp Response:", response)
        except Exception as e:
            print(f"llama-cpp test error: {e}")
    else:
        print(
            "llama-cpp is not available or model path not set - skipping test")
