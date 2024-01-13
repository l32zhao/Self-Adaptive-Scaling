# 使用官方的 MySQL 镜像作为基础镜像
FROM mysql:5.7

# 设置环境变量
ENV MYSQL_ROOT_PASSWORD=root

# 将 SQL 文件复制到容器内的初始化脚本目录
COPY ./user.sql /docker-entrypoint-initdb.d/init.sql

# 暴露 MySQL 的端口
EXPOSE 3306

# 定义 ENTRYPOINT 命令来启动 MySQL
ENTRYPOINT ["docker-entrypoint.sh"]

# 在 ENTRYPOINT 后添加 CMD 命令，以覆盖默认 CMD 命令
CMD ["mysqld"]
