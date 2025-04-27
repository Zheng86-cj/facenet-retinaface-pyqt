import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import sqlite3
import numpy as np
import pickle
import hashlib
import time

class DBManager:
    def __init__(self, db_path='data/face_system.db'):
        """初始化数据库管理器"""
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def connect(self):
        """连接到数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def disconnect(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def initialize_db(self):
        """初始化数据库表结构"""
        self.connect()
        
        # 创建用户表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0,
            register_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 检查并添加is_admin列（如果不存在）
        try:
            self.cursor.execute("SELECT is_admin FROM users LIMIT 1")
        except sqlite3.OperationalError:
            print("数据库升级: 添加is_admin列")
            self.cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
            self.conn.commit()
        
        # 检查并添加password列（如果不存在）
        try:
            self.cursor.execute("SELECT password FROM users LIMIT 1")
        except sqlite3.OperationalError:
            print("数据库升级: 添加password列")
            self.cursor.execute("ALTER TABLE users ADD COLUMN password TEXT")
            self.conn.commit()
        
        # 创建人脸特征表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            encoding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        ''')
        
        # 创建登录历史表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            login_method TEXT DEFAULT 'face',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        ''')
        
        # 检查并添加login_method列（如果不存在）
        try:
            self.cursor.execute("SELECT login_method FROM login_history LIMIT 1")
        except sqlite3.OperationalError:
            print("数据库升级: 添加login_method列")
            self.cursor.execute("ALTER TABLE login_history ADD COLUMN login_method TEXT DEFAULT 'face'")
            self.conn.commit()
        
        # 创建登录失败记录表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            failure_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attempt_method TEXT
        )
        ''')
        
        # 检查是否需要创建默认管理员账户
        self.cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        admin_count = self.cursor.fetchone()[0]
        
        if admin_count == 0:
            # 创建默认管理员账户 admin/admin123
            default_admin_password = self._hash_password("admin123")
            self.cursor.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
                ("admin", default_admin_password)
            )
            print("已创建默认管理员账户: 用户名=admin, 密码=admin123")
        
        self.conn.commit()
        self.disconnect()
    
    def _hash_password(self, password):
        """使用SHA-256哈希密码"""
        if not password:
            return None
        return hashlib.sha256(password.encode()).hexdigest()
    
    def add_user(self, username, password=None, is_admin=0):
        """添加新用户"""
        hashed_password = self._hash_password(password) if password else None
        
        self.connect()
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                (username, hashed_password, is_admin)
            )
            self.conn.commit()
            # 获取新插入用户的ID
            user_id = self.cursor.lastrowid
            return user_id
        except sqlite3.IntegrityError:
            # 用户名已存在
            return None
        finally:
            self.disconnect()
    
    def get_user_by_id(self, user_id):
        """通过ID获取用户信息"""
        self.connect()
        self.cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = self.cursor.fetchone()
        self.disconnect()
        return user
    
    def get_user_by_name(self, username):
        """通过用户名获取用户信息"""
        self.connect()
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = self.cursor.fetchone()
        self.disconnect()
        return user
    
    def verify_password(self, username, password):
        """验证用户的密码"""
        hashed_password = self._hash_password(password)
        self.connect()
        self.cursor.execute(
            "SELECT user_id, username, is_admin FROM users WHERE username = ? AND password = ?", 
            (username, hashed_password)
        )
        user = self.cursor.fetchone()
        self.disconnect()
        
        if user:
            return {"user_id": user[0], "username": user[1], "is_admin": user[2]}
        return None
    
    def get_all_users(self):
        """获取所有用户信息"""
        self.connect()
        self.cursor.execute("SELECT user_id, username, is_admin, register_time FROM users")
        users = self.cursor.fetchall()
        self.disconnect()
        return users
    
    def delete_user(self, user_id):
        """删除用户及其所有关联数据"""
        self.connect()
        try:
            # 删除用户的人脸特征
            self.cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
            
            # 删除用户的登录历史
            self.cursor.execute("DELETE FROM login_history WHERE user_id = ?", (user_id,))
            
            # 删除用户
            self.cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            
            self.conn.commit()
            return True
        except:
            return False
        finally:
            self.disconnect()
    
    def change_password(self, user_id, new_password):
        """修改用户密码"""
        hashed_password = self._hash_password(new_password)
        self.connect()
        try:
            self.cursor.execute(
                "UPDATE users SET password = ? WHERE user_id = ?", 
                (hashed_password, user_id)
            )
            self.conn.commit()
            return True
        except:
            return False
        finally:
            self.disconnect()
    
    def add_face_encoding(self, user_id, encoding):
        """添加人脸特征向量"""
        self.connect()
        # 将numpy数组序列化为二进制
        encoding_blob = pickle.dumps(encoding)
        self.cursor.execute("INSERT INTO face_encodings (user_id, encoding) VALUES (?, ?)", 
                           (user_id, sqlite3.Binary(encoding_blob)))
        self.conn.commit()
        self.disconnect()
    
    def get_face_encoding(self, user_id):
        """获取用户的人脸特征向量"""
        self.connect()
        self.cursor.execute("SELECT encoding FROM face_encodings WHERE user_id = ?", (user_id,))
        result = self.cursor.fetchone()
        self.disconnect()
        
        if result:
            # 将二进制反序列化为numpy数组
            return pickle.loads(result[0])
        return None
    
    def reset_face_encoding(self, user_id):
        """重置用户的人脸特征"""
        self.connect()
        try:
            self.cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
            self.conn.commit()
            return True
        except:
            return False
        finally:
            self.disconnect()
    
    def get_all_face_encodings(self):
        """获取所有用户的人脸特征向量"""
        self.connect()
        self.cursor.execute("""
            SELECT u.user_id, u.username, f.encoding 
            FROM users u 
            JOIN face_encodings f ON u.user_id = f.user_id
        """)
        results = self.cursor.fetchall()
        self.disconnect()
        
        encodings = {}
        for user_id, username, encoding_blob in results:
            encodings[user_id] = {
                'username': username,
                'encoding': pickle.loads(encoding_blob)
            }
        
        return encodings
    
    def record_login(self, user_id, method='face'):
        """记录用户登录"""
        self.connect()
        try:
            # 尝试使用login_method列记录登录
            self.cursor.execute("INSERT INTO login_history (user_id, login_method) VALUES (?, ?)", (user_id, method))
        except sqlite3.OperationalError as e:
            # 如果login_method列不存在，只记录用户ID
            if "no such column: login_method" in str(e):
                print("记录登录信息时使用兼容模式（无方法信息）")
                self.cursor.execute("INSERT INTO login_history (user_id) VALUES (?)", (user_id,))
            else:
                # 重新抛出其他错误
                raise e
        self.conn.commit()
        self.disconnect()
    
    def record_login_failure(self, username, method='face'):
        """记录登录失败"""
        self.connect()
        self.cursor.execute(
            "INSERT INTO login_failures (username, attempt_method) VALUES (?, ?)", 
            (username, method)
        )
        self.conn.commit()
        self.disconnect()
    
    def get_recent_failures(self, username, minutes=30):
        """获取用户最近的登录失败次数"""
        self.connect()
        timeframe = time.time() - (minutes * 60)  # 转换为秒
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeframe))
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM login_failures 
            WHERE username = ? AND failure_time > ?
        """, (username, formatted_time))
        
        count = self.cursor.fetchone()[0]
        self.disconnect()
        return count
    
    def get_login_history(self, user_id, limit=5):
        """获取用户登录历史"""
        self.connect()
        try:
            # 首先尝试获取带有login_method的历史记录
            self.cursor.execute("""
                SELECT login_time, login_method FROM login_history 
                WHERE user_id = ? 
                ORDER BY login_time DESC LIMIT ?
            """, (user_id, limit))
            results = self.cursor.fetchall()
        except sqlite3.OperationalError as e:
            # 如果login_method列不存在，只获取时间信息
            if "no such column: login_method" in str(e):
                print("获取登录历史记录时使用兼容模式（无方法信息）")
                self.cursor.execute("""
                    SELECT login_time, 'unknown' FROM login_history 
                    WHERE user_id = ? 
                    ORDER BY login_time DESC LIMIT ?
                """, (user_id, limit))
                results = self.cursor.fetchall()
            else:
                # 重新抛出其他错误
                raise
        self.disconnect()
        return results