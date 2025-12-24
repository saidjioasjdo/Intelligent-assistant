SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS=0;

DROP TABLE IF EXISTS `user`;
DROP TABLE IF EXISTS `chat_history`;
DROP TABLE IF EXISTS `history_summary`;

CREATE TABLE `user`(
    `uid` int(20) NOT NULL AUTO_INCREMENT,
    `name` varchar(30) NOT NULL,
    `password` varchar(40) NOT NULL,
    PRIMARY KEY (`uid`) USING BTREE
);

CREATE TABLE `chat_history`(
    `id` int(20) NOT NULL AUTO_INCREMENT UNIQUE KEY,
    `chat_id` int(20) NOT NULL DEFAULT 1,
    `AI_message` varchar(500) CHARACTER SET utf8mb4,
    `Human_message` varchar(500) CHARACTER SET utf8mb4,
    `uid` int(20) NOT NULL DEFAULT 1,
    PRIMARY KEY (`id`) USING BTREE,
    FOREIGN KEY (`uid`) REFERENCES `user`(`uid`)
);

CREATE TABLE `history_summary`(
    `id` int(20) NOT NULL AUTO_INCREMENT,
    `summary` varchar(100) CHARACTER SET utf8mb4,
    `uid` int(20) NOT NULL DEFAULT 1,
    `chat_id` int(20)  NOT NULL DEFAULT 1,
    PRIMARY KEY (`id`) USING BTREE,
    FOREIGN KEY (`uid`) REFERENCES `user`(`uid`)
);
SET FOREIGN_KEY_CHECKS = 1;