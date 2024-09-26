# -*- coding: utf-8 -*-
"""
@File:        schema.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
from marshmallow import fields, Schema, validate

from common.schema import SignSchema, CommResp
from common.constants import QueryType, ResponseType


class ChatSchema(SignSchema):
    """Chat schema"""

    conversation_id = fields.String(attribute="conversation_id", required=True)
    query = fields.String(attribute="query", required=True)
    conversation_type = fields.String(attribute="conversation_type", required=True,
                                      validate=validate.OneOf(
                                          [QueryType.NORMAL, QueryType.LINK, QueryType.REPORT, QueryType.DOC]))
    response_mode = fields.String(attribute="response_mode", required=True,
                                  validate=validate.OneOf([ResponseType.STREAM, ResponseType.BLOCK]))
    temperature = fields.Float(attribute="temperature", required=False, validate=validate.Range(0.0, 1.0))
    top_p = fields.Float(attribute="top_p", required=False, validate=validate.Range(0.0, 1.0))
