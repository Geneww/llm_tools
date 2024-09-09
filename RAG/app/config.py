# -*- coding: utf-8 -*-
"""
@File:        config.py
@Author:      Gene
@Software:    PyCharm
@Time:        09月 09, 2024
@Description:
"""
import datetime
import os
from typing import List, Type

basedir = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:
    CONFIG_NAME = "base"
    USE_MOCK_EQUIVALENCY = False
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


# 开发环境配置
class DevelopmentConfig(BaseConfig):
    CONFIG_NAME = "development"
    SECRET_KEY = os.getenv(
        "DEV_SECRET_KEY", "kxkkxkxkxxkxk"
    )
    DEBUG = True
    # 配置数据库的动态追踪修改
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_POOL_SIZE = 30
    SCHEDULER_API_ENABLED = True
    TESTING = False
    JSON_AS_ASCII = False
    # SQLALCHEMY_DATABASE_URI = os.getenv("DB_INFO", "mysql+pymysql://root:123456@172.16.40.14/itqm_aigc_db")
    SQLALCHEMY_DATABASE_URI = os.getenv("DB_INFO", "sqlite:///test.db")
    # jwt配置
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-4kDFdxxl2'
    JWT_COOKIE_CSRF_PROTECT = True
    JWT_CSRF_CHECK_FORM = True
    JWT_ACCESS_TOKEN_EXPIRES = os.environ.get('JWT_ACCESS_TOKEN_EXPIRES') or 36000
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(seconds=int(JWT_ACCESS_TOKEN_EXPIRES))
    PROPAGATE_EXCEPTIONS = True


# 生产环境配置
class ProductionConfig(BaseConfig):
    CONFIG_NAME = "production"
    SECRET_KEY = os.getenv("PROD_SECRET_KEY", "psk")
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = 30
    SCHEDULER_API_ENABLED = True
    TESTING = False
    JSON_AS_ASCII = False
    SQLALCHEMY_DATABASE_URI = os.getenv("DB_INFO", "mysql+pymysql://root:123456@localhost/db-xxxxx")
    # jwt配置
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-4kDFdxxl2'
    JWT_COOKIE_CSRF_PROTECT = True
    JWT_CSRF_CHECK_FORM = True
    JWT_ACCESS_TOKEN_EXPIRES = os.environ.get('JWT_ACCESS_TOKEN_EXPIRES') or 3600
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(seconds=int(JWT_ACCESS_TOKEN_EXPIRES))
    PROPAGATE_EXCEPTIONS = True


EXPORT_CONFIGS: List[Type[BaseConfig]] = [
    DevelopmentConfig,
    ProductionConfig,
]

config_by_name = {cfg.CONFIG_NAME: cfg for cfg in EXPORT_CONFIGS}

print(config_by_name)