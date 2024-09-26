# -*- coding: utf-8 -*-
"""
@File:        tools.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import uuid


def generate_uuid():
    return uuid.uuid4().hex


def generate_secret(length=16):
    import secrets
    # 生成一个16字节长度的随机秘钥
    secret_key = secrets.token_bytes(length)
    return secret_key.hex()


def build_prompt(prompt_template, **kwargs):
    """
    构建prompt模板
    :param prompt_template:
    :param kwargs:
    :return:
    """
    prompt = prompt_template
    if isinstance(prompt_template, list):
        content = kwargs.get("query")
        prompt_template.insert(-1, {"role": "user", "content": content})
    else:
        for k, v in kwargs.items():
            if isinstance(v, str):
                val = v
            elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
                val = '\n'.join(v)
            else:
                val = str(v)
            prompt = prompt_template.replace(f'__{k.upper()}__', val)
    return prompt


if __name__ == '__main__':
    print(generate_uuid())
    print(generate_secret())
    from config import *

    a = build_prompt(RAG_PROMPT_TEMPLATE, query="why sky blue?")
    b = build_prompt(PROMPT_TEMPLATE, query="why sky blue?")
    print(a)
    print(b)
