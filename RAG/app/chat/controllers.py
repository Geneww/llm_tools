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

from common.schema import custom_accepts
from common.response import json_response, RET
from app.chat.schema import ChatSchema, CommResp
from models.model import AppsManager

from app import logger

from services import ChatServices

api = Namespace("completion", description="LLM聊天api")


@api.route("/chat")
@api.doc(description="""
```
sign 签名方式: md5(app_id + nonce + timestamp + secret); 
nonce为 4位数字字符串(1000 - 9999)
```
---
```
query_type 为聊天类型 -> common: 通用聊天;  document: 专业知识问答; link: 功能定位聊天;
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
            logger.info('ret{a}'.format(a=ret))
            # 如果校验通过
            if ret:
                # 开始会话
                ChatServices.generator()

            return json_response(code=RET.OK, message="请求成功。")
        except Exception as e:
            logger.error(f"Completion request error: {e}")
            return json_response(code=400, message="请求失败：" + str(e))
