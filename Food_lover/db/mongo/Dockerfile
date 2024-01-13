# 使用官方的 MongoDB 镜像作为基础镜像
FROM mongo

# 设置环境变量
ENV MONGO_INITDB_DATABASE=fdlover
ENV MONGO_INITDB_ROOT_USERNAME=root
ENV MONGO_INITDB_ROOT_PASSWORD=root

# 将初始化脚本复制到容器内的初始化脚本目录
COPY ./mongo-init.js /docker-entrypoint-initdb.d/mongo-init.js

# 暴露 MongoDB 的端口
EXPOSE 27017

# 定义 ENTRYPOINT 命令来启动 MongoDB
ENTRYPOINT ["docker-entrypoint.sh"]

# 在 ENTRYPOINT 后添加 CMD 命令，以覆盖默认 CMD 命令
CMD ["mongod"]
