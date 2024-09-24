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
        db.Index('conversation_user_conversation_id_idx', 'user_conversation_id'),
    )

    user_conversation_id = db.Column(db.String(255), nullable=False)
    model_name = db.Column(db.String(255), nullable=False)
    conversation_type = db.Column(db.String(64), nullable=False)
    summary = db.Column(db.Text(), nullable=False)
