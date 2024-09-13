# -*- coding: utf-8 -*-
"""
@File:        response.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import time
import traceback
from functools import wraps

from flask import jsonify
from flask_accepts import accepts
from flask_restx import Namespace

from app import logger


def json_response(code=200, message="", data=None):
    return jsonify({
        "code": code,
        "message": message,
        "data": data,
        "timestamp": int(time.time())
    })


def custom_accepts(schema, api: Namespace):
    def decorator(func):
        @wraps(func)
        def default_accepts(*args, **kwargs):
            try:
                return accepts(schema=schema, api=api)(func)(*args, **kwargs)
            except Exception as e:
                logger.error(f"accepts error: {traceback.format_exc()}")
                return json_response(401, message="参数校验失败", data=str(e))

        return default_accepts

    return decorator



