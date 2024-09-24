# -*- coding: utf-8 -*-
"""
@File:        LLMbuilder.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 14, 2024
@Description:
"""
import os
from typing import Optional, List, Dict, Any, Mapping

from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult
from langchain_community.llms import Ollama

from core.llm.error import handle_llm_exceptions

# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "http://192.168.124.100:8000/v1/stream"
os.environ["OPENAI_API_KEY"] = "12acdwwww"


# class StreamableOpenAI(OpenAI):
#     # @handle_llm_exceptions
#     def generate(
#             self,
#             prompts: List[str],
#             stop: Optional[List[str]] = None,
#             callbacks: Callbacks = None,
#             **kwargs: Any,
#     ) -> LLMResult:
#         return super().generate(prompts, stop, callbacks, **kwargs)
#
#     @classmethod
#     def get_kwargs_from_model_params(cls, params: dict):
#         return params
def generate_prompt(prompt):
    messages = [
        {"role": "system",
         "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.


Example of a valid JSON response:
```json
{
    "title": "Initial Problem Analysis",
    "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.",
    "next_action": "continue"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant",
         "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]

    return messages


if __name__ == '__main__':
    user_query = """{
    "title": "Additional Factors: Considering Water Vapor and Ozone",
    "content": "Although we've looked into the impact of temperature changes on our visual perception, there are additional factors that must be taken into account. The presence and concentration of water vapor in the atmosphere plays a significant role in light absorption and scattering, influencing how colors appear to us. Additionally, ozone present in the upper layers can interact with certain wavelengths of light to affect its transmission through the atmosphere. To accurately understand these effects, we need to examine further how they impact our line of sight.",
    "next_action": "continue"
}"""
    prompt = generate_prompt(user_query)
    # sa = StreamableOpenAI()
    # print(sa.generate(prompts=[user_query]))
    llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-cot")  # 这里的"llama2"是你本地运行的模型名称
    response = llm.invoke(user_query)
    print(response)
