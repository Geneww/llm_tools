FROM python:3.10.11-slim-buster
LABEL authors="gene"

# 设置工作目录
WORKDIR /app

# 复制应用程序代码到容器中
COPY . /app

# 安装依赖项
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    pip install --upgrade pip -i https://pypi.douban.com/simple && \
    pip install -r requirements.txt -i https://pypi.douban.com/simple

# 设置环境变量
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV TZ=Asia/Shanghai

# 设置端口号
EXPOSE 8000

# 启动应用程序
CMD ["gunicorn", "-c", "wsgi_config.py", "wsgi:app"]