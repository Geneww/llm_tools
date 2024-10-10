# -*- coding: utf-8 -*-
"""
@File:        services.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
from typing import Union, Generator
from app import db, logger
from langchain_community.llms import Ollama

from .model import Conversation
from common.tools import build_prompt
from config import *


class ChatServices:
    @classmethod
    def chat(cls, conversation_id: str, query: str, conversation_type: str, stream: bool, temperature: float,
             top_p: float):
        llm = Ollama(base_url=LLM_MODEL_URL, model=LLM_MODEL)
        # 查询会话id是否存在，不存在就新建
        conversation = Conversation.query.filter_by(user_conversation_id=conversation_id).first()
        # 不存在则新建一个id
        if not conversation:
            conversation = Conversation(
                model_name=LLM_MODEL,
                user_conversation_id=conversation_id,
                conversation_type=conversation_type,
                summary=""
            )
            db.session.add(conversation)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logger.error("create new conversation model error.")
                raise e
        # 构建prompt
        prompt = build_prompt(PROMPT_TEMPLATE, query=query)
        # 调用LLM大模型
        if stream:
            response = llm.stream(prompt, temperature=temperature, top_p=top_p)
        else:
            response = llm.invoke(prompt, temperature=temperature, top_p=top_p)

        return response


if __name__ == '__main__':
    def test(name):
        llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-dpo")
        response = llm.invoke("为什么天空是蓝色的？", temperature=0.5, top_p=0.5)
        print(f"id:{name}- {response}")


    import threading

    for i in range(5):
        t = threading.Thread(target=test, kwargs={
            "name": i
        })
        t.start()
        print(t.name)
    print("done")
