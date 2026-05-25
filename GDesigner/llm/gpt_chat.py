import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization_key}",
    }

    data = {
        "model": model,
        "messages": msg,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            request_url,
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            text = await response.text()

            if response.status != 200:
                print("DeepSeek API Error")
                print("Status:", response.status)
                print("Content-Type:", response.headers.get("Content-Type"))
                print("Response:", text[:1000])
                raise RuntimeError(f"DeepSeek API failed with status {response.status}")

            try:
                response_data = await response.json()
            except Exception:
                print("Response is not JSON")
                print("Status:", response.status)
                print("Content-Type:", response.headers.get("Content-Type"))
                print("Response:", text[:1000])
                raise

            content = response_data["choices"][0]["message"]["content"]

            # DeepSeek 模型名不是 tiktoken 默认支持的 OpenAI 模型名，
            # 这里先不要调用 cost_count，否则可能因为 encoding_for_model 报错。
            # prompt = "".join([item["content"] for item in msg])
            # cost_count(prompt, content, model)

            return content
        

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass