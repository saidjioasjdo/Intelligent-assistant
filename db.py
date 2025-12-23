import pymysql
def connect_mysql():
    try:
        connect = pymysql.connect(
            host="localhost",
            user="root",
            password="123456789",
            database="chat",
            port=3306,
            charset='utf8mb4',
            autocommit=True
        )
        print("数据库连接成功")
        return connect
    except pymysql.Error as e:
        print(f"数据库链接失败：{e}")
        return None


if __name__ == '__main__':
    connect = connect_mysql()
    cur = connect.cursor()
    cur.execute("SELECT * FROM chat_history")
    # cur.execute("INSERT INTO user(name,password) VALUES('李四','1223');")
    d2=cur.fetchall()
    print(d2)
