create database fdlover;
use  fdlover;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL
);

INSERT INTO users (username, password) VALUES
    ('user1', 'password1hash'),  -- 请用密码的实际哈希代替 'password1hash'
    ('user2', 'password2hash'),
    ('user3', 'password3hash');