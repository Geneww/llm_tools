# -*- coding: utf-8 -*-
"""
@File:        schema.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 13, 2024
@Description:
"""
from marshmallow import fields, Schema


class LoginSchema(Schema):
    """login schema"""

    username = fields.String(attribute="username", required=True)
    password = fields.String(attribute="password", required=True)