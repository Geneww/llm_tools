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
from common.constants import ModelItem


class ChatServices:
    @classmethod
    def chat(cls, conversation_id: str, query: str, conversation_type: str, stream: bool):
        llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-cot")
        # 查询会话id是否存在，不存在就新建
        conversation = Conversation.query.filter_by(user_conversation_id=conversation_id).first()
        # 不存在则新建一个id
        if not conversation:
            conversation = Conversation(
                model_name=ModelItem.LLM,
                user_conversation_id=conversation_id,
                conversation_type=conversation_type,
                summary=""
            )
            db.session.add(conversation)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                raise e
        # 构建prompt
        if stream:
            response = llm.stream("hello!", temperature=0.7, top_p=0.9)
        else:
            response = llm.invoke("hello!", temperature=0.7, top_p=0.9)

        prompt = generate_prompt(query)
        # sa = StreamableOpenAI()
        # print(sa.generate(prompts=[user_query]))
        print(response)
        return {}


def test(name):
    llm = Ollama(base_url='http://192.168.124.100:11434', model="llama3-dpo")
    response = llm.invoke("为什么天空是蓝色的？", temperature=0.5, top_p=0.5)
    print(f"id:{name}- {response}")


if __name__ == '__main__':
    import threading

    for i in range(5):
        t = threading.Thread(target=test, kwargs={
            "name": i
        })
        t.start()
        print(t.name)
    print("done")
