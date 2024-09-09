# -*- coding: utf-8 -*-
"""
@File:        response.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import time
import json

from flask import jsonify


def json_response(code=200, message="", data=None):
    return jsonify({
        "code": code,
        "message": message,
        "data": data,
        "timestamp": int(time.time())
    })
