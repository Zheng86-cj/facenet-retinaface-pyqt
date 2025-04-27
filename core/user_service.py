import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import hashlib
import time
import datetime
import cv2
import numpy as np
from core.database_service import DBManager
from core.face_service import FaceManager

class UserManager:
    def __init__(self):
        """初始化用户管理器"""
        self.db_manager = DBManager()
        self.face_manager = FaceManager()
        
    def register_user(self, username, password=None, is_admin=0, camera_index=0, sample_count=5):
        """
        注册新用户
        
        Args:
            username: 用户名
            password: 可选的密码
            is_admin: 是否为管理员
            camera_index: 摄像头索引
            sample_count: 人脸样本数量
            
        Returns:
            成功返回用户ID，失败返回None
        """
        # 检查用户名是否已存在
        existing_user = self.db_manager.get_user_by_name(username)
        if existing_user:
            print(f"用户名 '{username}' 已存在")
            return None
        
        # 添加用户到数据库
        user_id = self.db_manager.add_user(username, password, is_admin)
        if user_id is None:
            print("用户添加失败")
            return None
            
        print(f"用户 '{username}' 注册成功，ID: {user_id}")
        return user_id
        
    def register_face(self, user_id, camera_index=0, sample_count=5):
        """
        为现有用户注册人脸
        
        Args:
            user_id: 用户ID
            camera_index: 摄像头索引
            sample_count: 人脸样本数量
            
        Returns:
            成功返回True，失败返回False
        """
        # 检查用户是否存在
        user = self.db_manager.get_user_by_id(user_id)
        if not user:
            print(f"用户ID {user_id} 不存在")
            return False
            
        username = user[1]
        
        # 采集人脸样本
        print(f"开始采集用户 '{username}' 的人脸样本...")
        face_encoding, sample_frames = self.face_manager.capture_face_sample(
            camera_index=camera_index, 
            sample_count=sample_count
        )
        
        if face_encoding is None:
            print("人脸采集失败，请重试")
            return False
        
        # 保存人脸特征
        self.db_manager.add_face_encoding(user_id, face_encoding)
        
        # 保存人脸样本图像到用户目录
        self._save_face_samples(username, sample_frames)
        
        print(f"用户 '{username}' 人脸注册成功")
        return True
    
    def _save_face_samples(self, username, sample_frames, max_samples=3):
        """保存人脸样本图像"""
        # 创建用户目录
        user_dir = os.path.join("face_samples", username)
        os.makedirs(user_dir, exist_ok=True)
        
        # 保存样本图像
        for i, frame in enumerate(sample_frames[:max_samples]):
            sample_path = os.path.join(user_dir, f"{username}_{i+1}.jpg")
            cv2.imwrite(sample_path, frame)
    
    def login_with_face(self, camera_index=0, max_attempts=3, timeout=30):
        """
        使用人脸登录
        
        Args:
            camera_index: 摄像头索引
            max_attempts: 最大尝试次数
            timeout: 超时时间(秒)
            
        Returns:
            成功返回用户信息(id, username)，失败返回None
        """
        # 获取所有用户的人脸特征
        known_encodings = self.db_manager.get_all_face_encodings()
        if not known_encodings:
            print("没有注册的用户")
            return None
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        
        start_time = time.time()
        attempts = 0
        
        while time.time() - start_time < timeout and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                print("无法从摄像头读取图像")
                break
            
            # 创建预览帧
            preview_frame = frame.copy()
            
            # 检测人脸
            _, faces = self.face_manager.detect_faces(frame, draw_result=False)
            
            # 如果检测到人脸
            if faces and len(faces) > 0:
                face = faces[0]
                x1, y1, x2, y2 = [int(p) for p in face[:4]]
                
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                face_encoding = self.face_manager.extract_face_encoding(frame, face[:4])
                
                if face_encoding is not None:
                    # 与数据库中的人脸比对
                    user_id, username, distance = self.face_manager.compare_faces(
                        known_encodings, face_encoding
                    )
                    
                    if user_id is not None:
                        # 登录成功
                        self.db_manager.record_login(user_id, 'face')
                        cv2.putText(preview_frame, f"欢迎回来, {username}!", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("人脸登录", preview_frame)
                        cv2.waitKey(1000)  # 显示识别结果1秒
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        print(f"登录成功，用户: {username}")
                        return user_id, username
                    else:
                        # 识别失败
                        cv2.putText(preview_frame, f"未识别, 相似度: {1-distance:.2f}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        attempts += 1
            
            cv2.imshow("人脸登录", preview_frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("登录失败，请重试")
        return None
    
    def login_with_password(self, username, password):
        """
        使用密码登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            成功返回用户信息，失败返回None
        """
        # 检查最近的登录失败次数
        recent_failures = self.db_manager.get_recent_failures(username)
        
        # 如果短时间内失败次数过多，则限制登录
        if recent_failures >= 5:
            print(f"由于多次登录失败，账户暂时锁定。请30分钟后再试")
            return None
        
        # 验证密码
        user = self.db_manager.verify_password(username, password)
        
        if user:
            # 登录成功
            self.db_manager.record_login(user["user_id"], 'password')
            print(f"密码登录成功，用户: {username}")
            return user
        else:
            # 登录失败，记录失败
            self.db_manager.record_login_failure(username, 'password')
            print("用户名或密码错误")
            return None
    
    def get_user_login_history(self, user_id, limit=5):
        """获取用户登录历史"""
        return self.db_manager.get_login_history(user_id, limit)
    
    def get_user_info(self, user_id):
        """获取用户信息"""
        user = self.db_manager.get_user_by_id(user_id)
        if user:
            # 返回包含字段名的字典，更清晰地表示各个字段
            return {
                "user_id": user[0],
                "username": user[1],
                "password": user[2],  # 注意：通常不应该传递密码
                "is_admin": user[3],
                "register_time": user[4]
            }
        return None
    
    def get_all_users(self):
        """获取所有用户信息"""
        return self.db_manager.get_all_users()
    
    def delete_user(self, user_id):
        """删除用户"""
        return self.db_manager.delete_user(user_id)
    
    def reset_user_face(self, user_id):
        """重置用户的人脸数据"""
        return self.db_manager.reset_face_encoding(user_id)
    
    def change_password(self, user_id, new_password):
        """修改用户密码"""
        return self.db_manager.change_password(user_id, new_password)
    
    def has_face_data(self, user_id):
        """检查用户是否有人脸数据"""
        face_encoding = self.db_manager.get_face_encoding(user_id)
        return face_encoding is not None 