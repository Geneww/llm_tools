# -*- coding: utf-8 -*-
"""
@File:        schema.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 13, 2024
@Description:
"""
from marshmallow import fields, Schema


class SignSchema(Schema):
    """Sign schema"""

    app_id = fields.String(attribute="app_id", required=True)
    timestamp = fields.Int(attribute="timestamp", required=True)
    nonce = fields.String(attribute="nonce", required=True)
    sign = fields.String(attribute="sign", required=True)


class CommResp(Schema):
    code = fields.Int(attribute="code", required=True)
    message = fields.String(attribute="message", required=True)
    data_list = fields.List(attribute="data", cls_or_instance=fields.List(cls_or_instance=fields.Int))
    data = fields.Dict(attribute="data_list", required=True)
    timestamp = fields.Int(attribute="timestamp", required=True)

