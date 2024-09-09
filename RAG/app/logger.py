# -*- coding: utf-8 -*-
"""
@File:        logger.py
@Author:      Gene
@Software:    PyCharm
@Time:        06月 05, 2024
@Description:
"""
import logging
import os
from logging import handlers


class Loggers:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        # 设置输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(levelname)s]-[%(filename)s]-[%(funcName)s:%(lineno)d] : %(message)s')
        # 定义一个日志收集器
        self.logger = logging.getLogger('log')
        # 设定级别
        self.logger.setLevel(logging.DEBUG)
        # 输出渠道一 - 文件形式
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        self.fileLogger = handlers.RotatingFileHandler("./logs/server.log", maxBytes=1024 * 1024 * 50, backupCount=3)
        # 输出渠道二 - 控制台
        self.console = logging.StreamHandler()
        # 控制台输出级别
        self.console.setLevel(logging.DEBUG)
        # 输出渠道对接输出格式
        self.console.setFormatter(formatter)
        self.fileLogger.setFormatter(formatter)
        # 日志收集器对接输出渠道
        self.logger.addHandler(self.fileLogger)
        self.logger.addHandler(self.console)

    def _log(self, level, msg, exc_info=None):
        # 使用 findCaller 方法来获取调用方的信息
        filename, lineno, funcname, sinfo = self.logger.findCaller(stack_info=False)
        if not self.logger.isEnabledFor(level):
            return
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn=filename,
            lno=lineno,
            msg=msg,
            args=(),
            exc_info=exc_info,
            func=funcname,
            extra=None
        )
        self.logger.handle(record)

    def debug(self, msg):
        self._log(logging.DEBUG, msg)

    def info(self, msg):
        self._log(logging.INFO, msg)

    def warn(self, msg):
        self._log(logging.WARNING, msg)

    def error(self, msg):
        self._log(logging.ERROR, msg)

    def exception(self, msg):
        self._log(logging.ERROR, msg, exc_info=True)


logger = Loggers()

# Example usage:
if __name__ == "__main__":
    logger = Loggers()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("An exception occurred")
