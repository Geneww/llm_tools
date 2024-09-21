# -*- coding: utf-8 -*-
"""
@File:        controllers.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 13, 2024
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

from common.response import json_response, custom_accepts
from app.system.schema import LoginSchema
from common.schema import CommResp
from models.model import AppsManager

from .. import logger

api = Namespace("system", description="LLM后台管理api")


@api.route("/login")
@api.doc(description="""sign 签名方式: md5(app_id + nonce + timestamp + secret); nonce为 4位数字字符串(1000 - 9999)
```
query_type 为聊天类型 -> common: 通用聊天; document: 专业知识问答; link: 功能定位聊天;
response_mode 为响应类型 -> block: 阻塞式一次性返回;  streaming: 流式返回;
```
""")
class Login(Resource):
    """登录api"""

    @custom_accepts(schema=LoginSchema, api=api)
    @responds(schema=CommResp, api=api)
    def post(self) -> Response:
        req_data = request.parsed_obj
        print(req_data)
        if not all([username, password]):
            return json_response(400, message="参数错误")
        return json_response(200, message="")
