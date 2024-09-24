# -*- coding: utf-8 -*-
"""
@File:        schema.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
from marshmallow import fields, Schema

from common.schema import SignSchema, CommResp


class ChatSchema(SignSchema):
    """Chat schema"""

    conversation_id = fields.String(attribute="conversation_id", required=True)
    query = fields.String(attribute="query", required=True)
    conversation_type = fields.String(attribute="conversation_type", required=True)
    response_mode = fields.String(attribute="response_mode", required=True)
