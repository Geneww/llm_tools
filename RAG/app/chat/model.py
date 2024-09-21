# -*- coding: utf-8 -*-
"""
@File:        model.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
from app import db
from models.model import BaseModel


class Conversation(db.Model, BaseModel):
    __tablename__ = 'conversations'
    __table_args__ = (
        db.Index('conversation_user_conv_id_idx', 'user_conv_id'),
    )

    conv_type = db.Column(db.String(64), nullable=False)
    model_id = db.Column(db.String(255), nullable=False)
    user_conv_id = db.Column(db.String(255), nullable=False)
    summary = db.Column(db.Text(), nullable=False)
