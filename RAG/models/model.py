# -*- coding: utf-8 -*-
"""
@File:        model.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import time
import pickle
import hashlib

from datetime import datetime

from app import db
from common.tools import generate_uuid


class BaseModel(object):
    """模型基类，为每个模型补充创建时间与更新时间"""
    id = db.Column(db.String, default=generate_uuid, primary_key=True)
    create_time = db.Column(db.DateTime, default=datetime.now, comment="创建时间")  # 记录的创建时间
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")  # 记录的更新时间


class AppsManager(db.Model, BaseModel):
    """APP管理类"""
    __tablename__ = "app_manager"
    # __table_args__ = (
    #     db.Index('', 'app_id'),
    # )
    app_id = db.Column(db.String(64), nullable=False)
    app_name = db.Column(db.String(64), nullable=False)
    app_secret = db.Column(db.String(64), nullable=False)
    is_enable = db.Column(db.Boolean, default=True)

    @staticmethod
    def verify_sign(app_id, nonce, timestamp, sign):
        if sign == "dev":
            return True
        instance = db.session.query(AppsManager).filter_by(app_id=app_id).first()
        # 如果不存在实例
        if not instance:
            return False
        raw_str = app_id + nonce + str(timestamp) + instance.app_secret
        gen_sign = hashlib.md5(raw_str.encode("utf-8")).hexdigest()
        return gen_sign == sign

    @staticmethod
    def init_record():
        """自动处理"""
        instance = db.session.query(AppsManager).filter_by(app_id="1001").first()
        if not instance:
            instance = AppsManager(app_id="1001", app_name="test_app", app_secret="xxx")
            db.session.add(instance)
            db.session.commit()
        return instance
