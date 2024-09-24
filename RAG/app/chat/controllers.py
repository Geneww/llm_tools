# -*- coding: utf-8 -*-
"""
@File:        controllers.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import json
import logging
import time
from typing import Union, Generator

from flask import request, stream_with_context
from flask_restx import Namespace, Resource
from flask.wrappers import Response
from flask_accepts import responds, accepts

from common.constants import QueryType
from common.schema import custom_accepts
from common.response import json_response, RET
from app.chat.schema import ChatSchema, CommResp
from models.model import AppsManager

from app import logger

from app.chat.services import ChatServices

api = Namespace("completion", description="LLM聊天api")


@api.route("/chat")
@api.doc(description="""
```
sign 签名方式: md5(app_id + nonce + timestamp + secret); 
nonce为 4位数字字符串(1000 - 9999)
```
---
```
conversation_type 为会话类型 -> common: 通用聊天;  document: 专业知识问答; link: 链接匹配; report: 报告总结；
response_mode 为响应类型 -> block: 阻塞式一次性返回;  
streaming: 流式返回;
```
""")
class Completion(Resource):
    @custom_accepts(schema=ChatSchema, api=api)
    @responds(schema=CommResp, api=api)
    def post(self) -> Response:
        try:
            req_data = request.parsed_obj
            print(req_data)
            ret = AppsManager.verify_sign(req_data["app_id"], req_data["nonce"], req_data["timestamp"],
                                          req_data["sign"])
            print(ret)
            # 权限校验
            if not ret:
                return json_response(code=RET.ROLEERR, message="权限错误")
            # 请求类型校验
            if req_data["conversation_type"] not in [QueryType.DOC, QueryType.NORMAL, QueryType.LINK, QueryType.REPORT]:
                return json_response(code=RET.PARAMERR, message="conversation_type错误")
            # 如果校验通过开始会话
            try:
                response = ChatServices.chat(
                    req_data["conversation_id"],
                    req_data["query"],
                    req_data["conversation_type"],
                    req_data["response_mode"]
                )
                return compact_response(response)
            except Exception as e:
                logger.error(e)

            return json_response(code=RET.OK, message="请求成功。")
        except Exception as e:
            logger.error(f"Completion request error: {e}")
            return json_response(code=400, message="请求失败：" + str(e))


def compact_response(response) -> Response:
    if isinstance(response, dict):
        return Response(response=json.dumps(response, ensure_ascii=False), mimetype="application/json")

    def generate() -> Generator:
        try:
            for chunk in response:
                yield chunk
        except Exception as e:
            logger.error(f"compact_response error: {e}")

    return Response(stream_with_context(generate()), status=200, mimetype="text/event-stream")
