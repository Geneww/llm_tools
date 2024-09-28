# -*- coding: utf-8 -*-
"""
@File:        response.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import time

from flask import jsonify


class RET:
    OK = "0"
    PARAMERR = "401"
    DATAERR = "402"
    USERERR = "403"
    ROLEERR = "404"
    REQERR = "405"
    SERVERERR = "406"


error_map = {
    RET.OK: u"请求成功",
    RET.PARAMERR: u"参数校验失败",
    RET.DATAERR: u"数据错误",
    RET.USERERR: u"用户不存在或未激活",
    RET.ROLEERR: u"用户身份错误",
    RET.REQERR: u"非法请求或请求次数受限",
    RET.SERVERERR: u"内部错误",
}


def json_response(code=200, message="", data=None):
    if message != "":
        msg = message
    else:
        msg = error_map[code]
    return jsonify({
        "code": code,
        "message": msg,
        "data": data,
        "timestamp": int(time.time())
    })


def message_response(code=200, message="", data=None):
    if message != "":
        msg = message
    else:
        msg = error_map[code]
    response_data = {
        "code": code,
        "message": msg,
        "data": data,
        "timestamp": int(time.time())
    }
    return response_data
