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
from typing import Union, Generator

from flask import request, stream_with_context
from flask_restx import Namespace, Resource
from flask.wrappers import Response
from flask_accepts import responds

from common.response import json_response

from .. import logger

api = Namespace("completion", description="LLM聊天api")


@api.route("/chat")
@api.doc(description="""sign 签名方式: md5(app_id + nonce + timestamp + secret); nonce为 4位数字字符串(1000 - 9999)
```
query_type 为聊天类型 -> common: 通用聊天; document: 专业知识问答; link: 功能定位聊天;
response_mode 为响应类型 -> block: 阻塞式一次性返回;  streaming: 流式返回;
```
""")
class Completion(Resource):
    def get(self) -> Response:
        try:
            pass
            return json_response(code=200, message="请求成功。")
        except Exception as e:
            logger.error(f"Completion request error: {e}")
            return json_response(code=400, message="请求失败：" + str(e))

    def post(self) -> Response:
        return json_response(code=200, message="123")