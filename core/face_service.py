import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import cv2
import numpy as np
import torch
from core.face_detector import Retinaface
from PIL import Image
from datetime import datetime
import time
import threading

class FaceManager:
    def __init__(self):
        """初始化人脸管理器"""
        # 初始化RetinaFace模型，用于人脸检测
        self.retinaface = Retinaface()
        # 人脸识别阈值
        self.recognition_threshold = 0.9
    
    def detect_faces(self, image, draw_result=True):
        """
        检测图像中的人脸
        
        Args:
            image: RGB格式的图像数组
            draw_result: 是否在图像上绘制检测结果
            
        Returns:
            处理后的图像，检测到的人脸框信息
        """
        # 备份原始图像
        original_image = image.copy()
        
        # 如果输入是BGR格式，转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                # 检查是否已经是RGB格式
                # OpenCV默认是BGR，但RetinaFace需要RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # 添加性能优化:
        # 1. 在低性能设备上，可以先进行人脸检测的图像缩放以加速处理
        # 计算输入图片的高和宽
        im_height, im_width = image_rgb.shape[:2]
        max_size = 320  # 检测时的最大尺寸
        
        # 计算缩放比例
        scale_factor = 1.0
        if max(im_height, im_width) > max_size:
            if im_width >= im_height:
                scale_factor = max_size / im_width
            else:
                scale_factor = max_size / im_height
                
            # 缩放图像
            detect_image = cv2.resize(image_rgb, None, fx=scale_factor, fy=scale_factor)
        else:
            detect_image = image_rgb
            
        # 进行人脸检测
        if draw_result:
            result_image = self.retinaface.detect_image(detect_image)
            
            # 如果进行了缩放，需要将结果图像缩放回原始尺寸
            if scale_factor < 1.0:
                result_image = cv2.resize(result_image, (im_width, im_height))
                
            detected_faces = self.retinaface.detected_faces
            
            # 如果进行了缩放，需要将检测到的人脸坐标恢复到原始尺寸
            if scale_factor < 1.0 and len(detected_faces) > 0:
                detected_faces = detected_faces.copy()
                for i, face in enumerate(detected_faces):
                    # 调整坐标
                    face[:4] = face[:4] / scale_factor
                    # 调整关键点坐标
                    face[5:] = face[5:] / scale_factor
                    detected_faces[i] = face
                    
                self.retinaface.detected_faces = detected_faces
                
            return result_image, detected_faces if hasattr(self.retinaface, 'detected_faces') else []
        else:
            # 这里对retinaface.py进行了修改，以便能够返回人脸位置而不是画在图像上
            # 我们需要自定义一个方法来获取人脸框和特征点
            detected_faces = self.retinaface.detect_image(detect_image, draw_bbox=False)
            
            # 如果进行了缩放，需要将检测到的人脸坐标恢复到原始尺寸
            if scale_factor < 1.0 and len(detected_faces) > 0:
                for i, face in enumerate(detected_faces):
                    # 调整坐标
                    face[:4] = face[:4] / scale_factor
                    # 调整关键点坐标
                    face[5:] = face[5:] / scale_factor
                    detected_faces[i] = face
                    
            self.extract_faces(image_rgb, detected_faces)
            return original_image, detected_faces
    
    def extract_faces(self, image, detected_faces=None):
        """
        从图像中提取人脸区域和特征点
        
        Args:
            image: RGB格式的图像数组
            detected_faces: 检测到的人脸框信息
            
        Returns:
            人脸区域和特征点信息
        """
        # 如果是BGR格式，转换为RGB (OpenCV默认是BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # 存储人脸数据
        if detected_faces is None:
            detected_faces = self.retinaface.detect_image(image_rgb, draw_bbox=False)
        self.retinaface.detected_faces = detected_faces
        
        return detected_faces
    
    def extract_face_encoding(self, image, face_location=None):
        """
        提取人脸特征向量
        
        Args:
            image: 图像数组
            face_location: 可选，人脸位置坐标(x1,y1,x2,y2)
            
        Returns:
            人脸特征向量
        """
        if face_location is None:
            # 如果没有提供人脸位置，则先进行人脸检测
            self.extract_faces(image)
            if not hasattr(self.retinaface, 'detected_faces') or len(self.retinaface.detected_faces) == 0:
                return None
            
            # 使用检测到的第一个人脸
            face_location = self.retinaface.detected_faces[0][:4]
        
        # 转换为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # 使用RetinaFace中的Facenet提取特征
        # 这需要对retinaface.py进行修改以暴露出facenet编码功能
        face_encoding = self.retinaface.get_face_encoding(image_rgb, face_location)
        
        return face_encoding
    
    def compare_faces(self, known_encodings, face_encoding, tolerance=None):
        """
        比较人脸特征向量
        
        Args:
            known_encodings: 已知人脸特征向量字典 {user_id: {'username': name, 'encoding': vector}}
            face_encoding: 待比对的人脸特征向量
            tolerance: 相似度阈值，越小越严格
            
        Returns:
            匹配的用户ID和用户名，如果没有匹配则返回None
        """
        if tolerance is None:
            tolerance = self.recognition_threshold
        
        best_match_id = None
        best_match_name = None
        min_distance = float('inf')
        
        for user_id, data in known_encodings.items():
            username = data['username']
            encoding = data['encoding']
            
            # 计算欧氏距离
            distance = np.linalg.norm(encoding - face_encoding)
            
            if distance < min_distance:
                min_distance = distance
                best_match_id = user_id
                best_match_name = username
        
        # 如果最小距离小于阈值，则认为匹配成功
        if min_distance < tolerance:
            return best_match_id, best_match_name, min_distance
        
        return None, None, min_distance
    
    def capture_face_sample(self, camera_index=0, sample_count=5):
        """
        从摄像头捕获多个人脸样本
        
        Args:
            camera_index: 摄像头索引
            sample_count: 采样数量
            
        Returns:
            人脸样本列表
        """
        # 尝试导入全局翻转设置
        try:
            from core.gui_controller import DO_FLIP, FLIP_MODE
            do_flip = DO_FLIP
            flip_mode = FLIP_MODE
        except ImportError:
            # 如果无法导入，使用默认设置
            do_flip = True
            flip_mode = -1
            
        cap = cv2.VideoCapture(camera_index)
        face_samples = []
        sample_frames = []
        
        while len(face_samples) < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 翻转图像
            if do_flip:
                frame = cv2.flip(frame, flip_mode)
                
            # 检测人脸
            _, faces = self.detect_faces(frame, draw_result=False)
            
            if faces and len(faces) > 0:
                # 只取检测到的第一个人脸
                face = faces[0]
                
                # 提取人脸特征
                face_encoding = self.extract_face_encoding(frame, face[:4])
                
                if face_encoding is not None:
                    face_samples.append(face_encoding)
                    sample_frames.append(frame.copy())
                    print(f"已采集 {len(face_samples)}/{sample_count} 个人脸样本")
            
            # 显示实时预览
            preview_frame = frame.copy()
            if faces and len(faces) > 0:
                face = faces[0]
                x1, y1, x2, y2 = [int(p) for p in face[:4]]
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            cv2.imshow('人脸采集', preview_frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 计算平均特征向量
        if face_samples:
            average_encoding = np.mean(face_samples, axis=0)
            return average_encoding, sample_frames
        
        return None, [] 