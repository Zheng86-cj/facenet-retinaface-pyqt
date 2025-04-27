import os
import sys

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 定义数据目录
DATA_DIR = os.path.join(root_dir, 'data')
DB_PATH = os.path.join(DATA_DIR, 'face_system.db')
FACE_SAMPLES_DIR = os.path.join(DATA_DIR, 'face_samples')
BACKUP_DIR = os.path.join(DATA_DIR, 'backups')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

# 确保必要的目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_SAMPLES_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 原始导入部分
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import atexit
import time
import threading
from datetime import datetime

# 修改内部导入
from core.user_service import UserManager
from core.face_service import FaceManager
from core.database_service import DBManager

# 全局配置参数
DO_FLIP = True         # 是否进行图像翻转
FLIP_MODE = -1          # 翻转模式: 0=上下翻转, 1=左右翻转, -1=上下左右翻转
LOW_PERFORMANCE = False # 是否使用低性能模式
MAX_PROCESS_WIDTH = 640 # 处理图像的最大宽度

# 全局变量存储摄像头对象
active_cameras = []

# 全局锁，用于保护摄像头访问
camera_lock = QMutex()

def put_chinese_text(img, text, position, color=(0, 255, 0), size=20):
    """在图像上绘制中文文本
    Args:
        img: OpenCV图像
        text: 要绘制的文本
        position: 文本位置 (x, y)
        color: 文本颜色
        size: 字体大小
    Returns:
        添加文字后的图像
    """
    if not text:  # 如果文本为空，直接返回原图
        return img
    
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(img_pil)
    
    # 加载字体，使用model_data目录下的simhei.ttf字体
    try:
        font = ImageFont.truetype("model_data/simhei.ttf", size)  # 使用项目中的黑体字体
    except Exception as e:
        print(f"加载字体失败: {e}，尝试使用系统字体")
        try:
            font = ImageFont.truetype("simhei.ttf", size)  # 尝试使用系统黑体
        except:
            try:
                font = ImageFont.truetype("simsun.ttc", size)  # 尝试使用宋体
            except:
                font = ImageFont.load_default()  # 使用默认字体
        
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 将PIL图像转换回OpenCV格式
    img_opencv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_opencv

def release_all_cameras():
    """释放所有打开的摄像头资源"""
    global active_cameras, camera_lock
    camera_lock.lock()
    try:
        for cap in active_cameras:
            if cap and cap.isOpened():
                cap.release()
        active_cameras = []
        print("所有摄像头资源已释放")
    finally:
        camera_lock.unlock()

# 注册程序退出时自动释放摄像头资源
atexit.register(release_all_cameras)

class VideoWidget(QLabel):
    """视频显示控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        # 使用更简单的样式，避免样式冲突
        self.setStyleSheet("background-color: #222222;")
        # 设置白色文本
        self.setText("等待摄像头画面...")
        self.setFont(QFont("Arial", 14))
        palette = self.palette()
        palette.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(palette)

    def display_frame(self, frame):
        """显示帧图像"""
        if frame is None:
            self.setText("无法获取摄像头画面")
            return
        
        try:
            # 确保图像有正确的尺寸和通道
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.setText(f"图像格式错误: {frame.shape}")
                return
                
            # 转换BGR到RGB格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # 使用RGB格式
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放图像，确保不超过标签大小
            label_size = self.size()
            scaled_pixmap = QPixmap.fromImage(convert_to_qt_format).scaled(
                label_size.width(), label_size.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 设置图像并清除文本
            self.setPixmap(scaled_pixmap)
            self.setText("")
            
        except Exception as e:
            import traceback
            self.setText(f"显示图像出错: {str(e)}")
            traceback.print_exc()
            

class RegisterDialog(QDialog):
    """用户注册对话框"""
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.camera_index = 0
        self.sample_count = 5
        self.collected_samples = 0
        self.cap = None
        self.user_id = None  # 存储注册的用户ID
        
        self.setWindowTitle("用户注册")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QLabel { font-size: 14px; }
            QLineEdit { 
                height: 30px; 
                border-radius: 4px; 
                border: 1px solid #ccc; 
                padding: 5px; 
                font-size: 14px; 
            }
            QPushButton { 
                height: 35px; 
                border-radius: 4px; 
                background-color: #4CAF50; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 选项卡控件
        self.tabs = QTabWidget()
        
        # 账户信息选项卡
        account_tab = self.create_account_tab()
        self.tabs.addTab(account_tab, "账户信息")
        
        # 人脸采集选项卡
        face_tab = self.create_face_tab()
        self.tabs.addTab(face_tab, "人脸采集")
        self.tabs.setTabEnabled(1, False)  # 初始禁用人脸采集选项卡
        
        main_layout.addWidget(self.tabs)
        
        # 状态标签
        self.status_label = QLabel("请先创建账户")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("background-color: #f44336;")
        
        self.next_button = QPushButton("创建账户")
        self.next_button.clicked.connect(self.create_account)
        
        self.finish_button = QPushButton("完成注册")
        self.finish_button.clicked.connect(self.finish_registration)
        self.finish_button.setEnabled(False)
        
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.next_button)
        buttons_layout.addWidget(self.finish_button)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
    
    def create_account_tab(self):
        """创建账户信息选项卡"""
        account_tab = QWidget()
        layout = QVBoxLayout()
        
        # 用户信息组
        info_group = QGroupBox("用户信息")
        info_layout = QGridLayout()
        
        # 用户名
        username_label = QLabel("用户名:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        info_layout.addWidget(username_label, 0, 0)
        info_layout.addWidget(self.username_input, 0, 1)
        
        # 密码
        password_label = QLabel("密码:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码（可选）")
        self.password_input.setEchoMode(QLineEdit.Password)
        info_layout.addWidget(password_label, 1, 0)
        info_layout.addWidget(self.password_input, 1, 1)
        
        # 确认密码
        confirm_label = QLabel("确认密码:")
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        info_layout.addWidget(confirm_label, 2, 0)
        info_layout.addWidget(self.confirm_input, 2, 1)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 说明文字
        note_label = QLabel("注意: 您可以选择设置密码或稍后仅使用人脸登录。")
        note_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(note_label)
        
        account_tab.setLayout(layout)
        return account_tab
    
    def create_face_tab(self):
        """创建人脸采集选项卡"""
        face_tab = QWidget()
        layout = QVBoxLayout()
        
        # 摄像头预览
        self.video_frame = VideoWidget()
        layout.addWidget(self.video_frame)
        
        # 人脸采集进度条
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("采集进度: 0/5")
        progress_layout.addWidget(self.progress_label)
        
        # 采集进度可视化
        self.progress_frames = []
        for i in range(5):
            frame = QFrame()
            frame.setFixedSize(50, 50)
            frame.setStyleSheet("background-color: #ddd; border-radius: 25px;")
            self.progress_frames.append(frame)
            progress_layout.addWidget(frame)
        
        progress_layout.addStretch()
        layout.addLayout(progress_layout)
        
        # 采集控制按钮
        control_layout = QHBoxLayout()
        
        self.start_capture_button = QPushButton("开始采集")
        self.start_capture_button.clicked.connect(self.start_capture)
        
        self.skip_button = QPushButton("跳过")
        self.skip_button.clicked.connect(self.skip_face_capture)
        self.skip_button.setStyleSheet("background-color: #FF9800;")
        
        control_layout.addWidget(self.start_capture_button)
        control_layout.addWidget(self.skip_button)
        
        layout.addLayout(control_layout)
        
        face_tab.setLayout(layout)
        return face_tab
    
    def create_account(self):
        """创建用户账户"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        
        # 验证输入
        if not username:
            QMessageBox.warning(self, "输入错误", "请输入用户名")
            return
        
        if password and password != confirm:
            QMessageBox.warning(self, "密码不匹配", "两次输入的密码不一致")
            return
        
        # 检查用户名是否已存在
        existing_user = self.user_manager.db_manager.get_user_by_name(username)
        if existing_user:
            QMessageBox.warning(self, "用户已存在", f"用户名 '{username}' 已存在")
            return
        
        # 创建用户账户
        self.user_id = self.user_manager.register_user(username, password)
        
        if self.user_id:
            self.status_label.setText(f"账户创建成功! 用户ID: {self.user_id}")
            
            # 禁用账户表单并启用人脸采集选项卡
            self.username_input.setEnabled(False)
            self.password_input.setEnabled(False)
            self.confirm_input.setEnabled(False)
            
            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            
            # 更新按钮状态
            self.next_button.setEnabled(False)
            self.finish_button.setEnabled(True)
            
            # 如果未设置密码，提示用户必须完成人脸采集
            if not password:
                QMessageBox.information(self, "提示", "您没有设置密码，请完成人脸采集以便将来登录")
                self.skip_button.setEnabled(False)
        else:
            QMessageBox.critical(self, "账户创建失败", "无法创建用户账户")
    
    def start_capture(self):
        """开始人脸采集"""
        if self.user_id is None:
            QMessageBox.warning(self, "错误", "请先创建账户")
            return
        
        # 切换UI状态
        self.start_capture_button.setText("注册")
        self.start_capture_button.clicked.disconnect()
        self.start_capture_button.clicked.connect(self.register_face)
        
        # 初始化前关闭可能存在的资源
        if hasattr(self, 'timer') and self.timer and self.timer.isActive():
            self.timer.stop()
            
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            if self.cap in active_cameras:
                active_cameras.remove(self.cap)
                
        # 设置视频显示控件提示
        self.video_frame.setText("正在打开摄像头...")
        
        try:
            print("=== RegisterDialog: 摄像头初始化开始 ===")
            print(f"尝试打开摄像头 (索引: {self.camera_index})...")
            
            # 使用最简单的方式初始化摄像头
            print("使用最简单的初始化方式")
            self.cap = cv2.VideoCapture(self.camera_index)
            print(f"摄像头对象创建: {self.cap}")
            
            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                print(f"错误: 无法打开摄像头 (索引: {self.camera_index})")
                print("请检查:")
                print("1. 摄像头是否正确连接")
                print("2. 摄像头驱动是否正确安装")
                print("3. 是否给予应用摄像头使用权限")
                print("4. 是否有其他应用正在使用摄像头")
                
                self.video_frame.setText(f"无法打开摄像头 (索引: {self.camera_index})")
                QMessageBox.critical(self, "错误", f"无法打开摄像头 (索引: {self.camera_index})，请检查设备连接")
                self.start_capture_button.setText("开始采集")
                self.start_capture_button.clicked.disconnect()
                self.start_capture_button.clicked.connect(self.start_capture)
                return
            
            # 摄像头已成功打开，设置参数
            print("摄像头已成功打开，设置参数...")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"摄像头参数设置完成")
            print(f"分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
            
            # 添加到全局摄像头列表中
            active_cameras.append(self.cap)
            print(f"摄像头已添加到全局列表，当前列表大小: {len(active_cameras)}")
            
            # 多次尝试读取第一帧，使用更长的超时
            print("尝试读取第一帧...")
            frame = None
            for attempt in range(1, 10):  # 增加到10次尝试
                print(f"读取帧尝试 {attempt}/10")
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"成功读取帧，尺寸: {frame.shape}")
                    break
                print("读取失败，等待0.3秒后重试...")
                time.sleep(0.3)  # 增加延迟
                
            if frame is None:
                print("无法读取摄像头图像，所有尝试都失败")
                self.video_frame.setText("摄像头无法读取图像")
                QMessageBox.critical(self, "错误", "摄像头无法读取图像，请检查摄像头权限或驱动")
                self.cap.release()
                self.cap = None
                if self.cap in active_cameras:
                    active_cameras.remove(self.cap)
                self.start_capture_button.setText("开始采集")
                self.start_capture_button.clicked.disconnect()
                self.start_capture_button.clicked.connect(self.start_capture)
                return
                
            # 确认第一帧可以读取
            if DO_FLIP:
                frame = cv2.flip(frame, FLIP_MODE)
                
            # 保存第一帧用于调试
            try:
                debug_frame_path = os.path.join(TEMP_DIR, "register_first_frame.jpg")
                cv2.imwrite(debug_frame_path, frame)
                print("已保存调试帧到", debug_frame_path)
            except Exception as e:
                print(f"保存调试帧失败: {e}")
            
            # 在视频控件中显示第一帧
            self.video_frame.display_frame(frame)
            
            # 开始定时器更新视频帧
            print("创建和启动定时器...")
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30fps
            print("定时器已启动，摄像头初始化完成")
            print("=== RegisterDialog: 摄像头初始化结束 ===")
            
        except Exception as e:
            print(f"摄像头初始化错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.video_frame.setText(f"摄像头初始化错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"摄像头初始化错误: {str(e)}")
            self.start_capture_button.setText("开始采集")
            self.start_capture_button.clicked.disconnect()
            self.start_capture_button.clicked.connect(self.start_capture)
            return
        
        self.collected_samples = 0
        self.collected_encodings = []
        self.collected_frames = []
        
        # 更新UI状态
        self.status_label.setText("请正视摄像头，开始采集人脸样本")
        self.skip_button.setEnabled(False)
    
    def register_face(self):
        """注册人脸特征"""
        if self.collected_samples < self.sample_count:
            QMessageBox.warning(self, "采集未完成", "请先完成人脸采集")
            return
        
        # 计算平均特征向量
        average_encoding = np.mean(self.collected_encodings, axis=0)
        
        # 获取用户名
        user = self.user_manager.get_user_info(self.user_id)
        username = user.get('username', '未知用户') if isinstance(user, dict) else "未知用户"
        
        # 保存人脸特征
        self.user_manager.db_manager.add_face_encoding(self.user_id, average_encoding)
        
        # 保存人脸样本图像
        self.user_manager._save_face_samples(username, self.collected_frames)
        
        QMessageBox.information(self, "注册成功", f"人脸特征注册成功")
        
        # 停止摄像头和计时器
        self.stop_camera()
        
        self.accept()
    
    def skip_face_capture(self):
        """跳过人脸采集"""
        # 如果用户设置了密码，允许跳过人脸采集
        if not self.password_input.text():
            QMessageBox.warning(self, "错误", "您没有设置密码，必须完成人脸采集")
            return
        
        reply = QMessageBox.question(
            self, 
            "确认跳过", 
            "确定要跳过人脸采集吗？您将只能使用密码登录。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.accept()
    
    def finish_registration(self):
        """完成注册"""
        # 如果用户没有设置密码且没有采集人脸，提示错误
        has_password = bool(self.password_input.text())
        has_face = self.collected_samples >= self.sample_count
        
        if not has_password and not has_face:
            QMessageBox.warning(self, "注册错误", "您必须设置密码或完成人脸采集")
            return
        
        if has_face and self.collected_samples >= self.sample_count:
            # 如果已经采集了足够的人脸样本，保存特征
            self.register_face()
        else:
            # 如果有密码但没有完成人脸采集，直接完成注册
            self.accept()
    
    def update_frame(self):
        """更新视频帧"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 使用全局配置参数翻转图像
        if DO_FLIP:
            frame = cv2.flip(frame, FLIP_MODE)
        
        # 缩小图像以提高处理速度
        frame_height, frame_width = frame.shape[:2]
        if frame_width > MAX_PROCESS_WIDTH:
            scale_factor = MAX_PROCESS_WIDTH / frame_width
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        
        # 备份原始帧用于预览
        preview_frame = frame.copy()
        faces = []
            
        # 使用人脸管理器检测人脸
        try:
            print("尝试检测人脸...")
            _, detected_faces = self.user_manager.face_manager.detect_faces(frame, draw_result=False)
            
            # 确保检测的结果是有效的
            if detected_faces is not None and len(detected_faces) > 0:
                print(f"检测到 {len(detected_faces)} 个人脸")
                faces = detected_faces
                
                # 在预览帧上绘制人脸框
                for face in faces:
                    if len(face) >= 4:  # 确保有足够的坐标
                        x1, y1, x2, y2 = [int(p) for p in face[:4]]
                        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                print("未检测到人脸")
                # 使用中文文本绘制函数
                preview_frame = put_chinese_text(preview_frame, "请正视摄像头", (20, 40), color=(0, 255, 255), size=24)
        except Exception as e:
            # 发生错误时记录错误，但继续显示原始帧
            print(f"人脸检测错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 在预览帧上显示错误信息
            preview_frame = put_chinese_text(preview_frame, "检测错误", (20, 40), color=(0, 0, 255), size=24)
        
        # 自动采集样本
        if len(faces) > 0 and self.collected_samples < self.sample_count:
            # 限制采样频率，每1秒采集一次
            if not hasattr(self, 'last_capture_time') or time.time() - self.last_capture_time > 1:
                try:
                    print("尝试进行人脸采集...")
                    face = faces[0]
                    face_encoding = self.user_manager.face_manager.extract_face_encoding(frame, face[:4])
                    
                    if face_encoding is not None:
                        self.collected_encodings.append(face_encoding)
                        self.collected_frames.append(frame.copy())
                        self.collected_samples += 1
                        print(f"成功采集第 {self.collected_samples} 个样本")
                        
                        # 更新进度
                        self.progress_label.setText(f"采集进度: {self.collected_samples}/{self.sample_count}")
                        
                        # 更新进度可视化
                        for i in range(self.sample_count):
                            if i < self.collected_samples:
                                self.progress_frames[i].setStyleSheet("background-color: #4CAF50; border-radius: 25px;")
                            else:
                                self.progress_frames[i].setStyleSheet("background-color: #ddd; border-radius: 25px;")
                        
                        # 根据采集进度更新提示
                        if self.collected_samples == 1:
                            self.status_label.setText("请向左转头...")
                        elif self.collected_samples == 2:
                            self.status_label.setText("请向右转头...")
                        elif self.collected_samples == 3:
                            self.status_label.setText("请抬头...")
                        elif self.collected_samples == 4:
                            self.status_label.setText("请低头...")
                        elif self.collected_samples == 5:
                            self.status_label.setText("采集完成，请点击 '注册' 按钮")
                            
                        # 如果采集完成，启用注册按钮
                        if self.collected_samples >= self.sample_count:
                            self.finish_button.setEnabled(True)
                    else:
                        print("提取人脸特征失败")
                    
                    self.last_capture_time = time.time()
                except Exception as e:
                    print(f"特征提取错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # 显示采集状态在预览画面上
        preview_frame = put_chinese_text(
            preview_frame, 
            f"采集进度: {self.collected_samples}/{self.sample_count}", 
            (20, 80), 
            color=(255, 255, 255), 
            size=24
        )
                   
        # 显示预览
        self.video_frame.display_frame(preview_frame)
    
    def stop_camera(self):
        """停止摄像头和计时器"""
        if hasattr(self, 'timer') and self.timer and self.timer.isActive():
            self.timer.stop()
            print("RegisterDialog: 定时器已停止")
        
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            if self.cap in active_cameras:
                active_cameras.remove(self.cap)
                print("RegisterDialog: 摄像头已从全局列表中移除")
            self.cap = None
            print("RegisterDialog: 摄像头已释放")
    
    def closeEvent(self, event):
        """窗口关闭事件，释放资源"""
        print("关闭RegisterDialog，释放资源...")
        self.stop_camera()
        print("RegisterDialog资源释放完成")
        event.accept()


class CameraThread(QThread):
    """摄像头线程，避免UI阻塞"""
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    initialized = pyqtSignal(bool)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        self.frame_count = 0
        print(f"CameraThread初始化: 索引={camera_index}")
    
    def run(self):
        """线程运行函数"""
        global camera_lock
        try:
            print(f"摄像头线程: 开始运行")
            print(f"摄像头线程: 尝试获取摄像头锁...")
            
            camera_lock.lock()
            try:
                print(f"摄像头线程: 尝试打开摄像头 (索引: {self.camera_index})...")
                # 多种方式尝试打开摄像头
                for i in range(3):
                    try:
                        if i == 0:
                            print("尝试方法1: 默认方式")
                            self.cap = cv2.VideoCapture(self.camera_index)
                        elif i == 1:
                            print("尝试方法2: 指定后端方式")
                            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                        else:
                            print("尝试方法3: 重新连接")
                            if self.cap:
                                self.cap.release()
                            time.sleep(1.0)  # 等待系统释放资源
                            self.cap = cv2.VideoCapture(self.camera_index)
                        
                        if self.cap and self.cap.isOpened():
                            print(f"摄像头打开成功，方法 {i+1}")
                            break
                        else:
                            print(f"方法 {i+1} 失败，尝试下一个方法")
                            time.sleep(0.5)
                    except Exception as e:
                        print(f"摄像头初始化错误 (方法 {i+1}): {str(e)}")
                
                if not self.cap or not self.cap.isOpened():
                    print(f"摄像头线程: 无法打开摄像头 (索引: {self.camera_index})")
                    self.error.emit(f"无法打开摄像头 (索引: {self.camera_index})")
                    self.initialized.emit(False)
                    return
                
                # 设置摄像头参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                print(f"摄像头线程: 摄像头参数设置完成")
                print(f"摄像头线程: 分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                print(f"摄像头线程: FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                
                # 读取第一帧测试
                print("摄像头线程: 尝试读取第一帧...")
                frame_read = False
                
                for attempt in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_read = True
                        print(f"摄像头线程: 成功读取第一帧，尺寸: {frame.shape}，尝试次数: {attempt+1}")
                        break
                    print(f"摄像头线程: 读取失败 ({attempt+1}/5)，等待0.5秒...")
                    time.sleep(0.5)
                
                if not frame_read:
                    print("摄像头线程: 无法读取摄像头图像")
                    self.error.emit("无法读取摄像头图像")
                    self.initialized.emit(False)
                    return
                
                # 保存第一帧用于调试
                try:
                    debug_frame_path = os.path.join(TEMP_DIR, "thread_first_frame.jpg")
                    cv2.imwrite(debug_frame_path, frame)
                    print("摄像头线程: 已保存调试帧到", debug_frame_path)
                except Exception as e:
                    print(f"摄像头线程: 保存调试帧失败: {e}")
                    
                print(f"摄像头线程: 成功初始化，分辨率: {frame.shape[1]}x{frame.shape[0]}")
                
                # 保存到全局列表
                global active_cameras
                if self.cap not in active_cameras:
                    active_cameras.append(self.cap)
                    print(f"摄像头线程: 摄像头已添加到全局列表，当前列表大小: {len(active_cameras)}")
            finally:
                camera_lock.unlock()
            
            # 发送初始化成功信号
            print("摄像头线程: 发送初始化成功信号")
            self.initialized.emit(True)
            
            # 主循环
            self.running = True
            self.frame_count = 0
            print("摄像头线程: 进入主循环")
            last_time = time.time()
            
            while self.running:
                if not self.cap.isOpened():
                    print("摄像头线程: 摄像头已关闭")
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("摄像头线程: 无法读取帧")
                    if self.frame_count > 0:  # 只有在之前成功过的情况下重试
                        time.sleep(0.1)
                        continue
                    else:
                        break  # 从未成功过则退出
                
                # 翻转图像
                if DO_FLIP:
                    frame = cv2.flip(frame, FLIP_MODE)
                
                # 发送帧到UI线程
                self.frame_ready.emit(frame.copy())  # 发送副本避免引用问题
                self.frame_count += 1
                
                # 定期打印状态
                if time.time() - last_time > 5.0:  # 每5秒报告一次
                    print(f"摄像头线程: 已处理 {self.frame_count} 帧，FPS: {self.frame_count / (time.time() - last_time):.1f}")
                    last_time = time.time()
                    self.frame_count = 0
                
                # 控制帧率
                time.sleep(0.03)  # 约30fps
            
            print("摄像头线程: 主循环结束")
        
        except Exception as e:
            print(f"摄像头线程: 异常 - {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"摄像头错误: {str(e)}")
            self.initialized.emit(False)
        
        finally:
            self.running = False
            camera_lock.lock()
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    if self.cap in active_cameras:
                        active_cameras.remove(self.cap)
                        print(f"摄像头线程: 摄像头已从全局列表中移除，当前列表大小: {len(active_cameras)}")
            finally:
                camera_lock.unlock()
            print("摄像头线程: 已结束")
    
    def stop(self):
        """停止线程"""
        print("摄像头线程: 停止请求")
        self.running = False
        if not self.wait(5000):  # 等待最多5秒
            print("摄像头线程: 等待超时，强制终止")
            self.terminate()
        print("摄像头线程: 已停止")


class LoginWidget(QWidget):
    """人脸登录界面"""
    login_successful = pyqtSignal(int, str, bool)  # 添加管理员标志
    
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        print("初始化LoginWidget...")
        self.user_manager = user_manager
        self.camera_index = 0
        self.cap = None
        self.timer = None
        self.camera_thread = None
        self.current_frame = None
        
        self.init_ui()
        print("LoginWidget初始化完成")
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("人脸识别登录系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        main_layout.addWidget(title_label)
        
        # 视频预览
        self.video_frame = VideoWidget()
        main_layout.addWidget(self.video_frame)
        
        # 状态提示
        self.status_label = QLabel("请面对摄像头进行人脸识别")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; margin: 10px;")
        main_layout.addWidget(self.status_label)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        self.face_login_button = QPushButton("人脸登录")
        self.face_login_button.setFixedHeight(40)
        self.face_login_button.clicked.connect(self.start_face_login)
        
        self.password_login_button = QPushButton("密码登录")
        self.password_login_button.setFixedHeight(40)
        self.password_login_button.clicked.connect(self.start_password_login)
        self.password_login_button.setStyleSheet("background-color: #FF9800;")
        
        self.register_button = QPushButton("注册新用户")
        self.register_button.setFixedHeight(40)
        self.register_button.clicked.connect(self.open_register_dialog)
        
        buttons_layout.addWidget(self.face_login_button)
        buttons_layout.addWidget(self.password_login_button)
        buttons_layout.addWidget(self.register_button)
        
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)
    
    def start_face_login(self):
        """开始人脸登录流程"""
        # 检查是否有注册用户
        known_encodings = self.user_manager.db_manager.get_all_face_encodings()
        if not known_encodings:
            QMessageBox.warning(self, "无用户", "系统中没有注册的用户，请先注册")
            return
        
        # 修改按钮状态
        self.face_login_button.setText("正在识别...")
        self.face_login_button.setEnabled(False)
        
        # 确保关闭之前的摄像头和定时器
        self.stop_camera()
        
        # 延迟一下，确保之前的摄像头已完全释放
        QApplication.processEvents()
        time.sleep(0.5)
        
        # 设置视频显示控件提示
        self.video_frame.setText("正在打开摄像头...")
        self.status_label.setText("正在打开摄像头，请稍候...")
        
        # 强制处理UI事件，确保提示显示
        QApplication.processEvents()
        
        # 在独立线程中打开摄像头
        print("=== 摄像头初始化开始 ===")
        print("创建摄像头线程...")
        self.camera_thread = CameraThread(self.camera_index)
        
        print("连接摄像头信号...")
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error.connect(self.on_camera_error)
        self.camera_thread.initialized.connect(self.on_camera_initialized)
        
        print("启动摄像头线程...")
        self.camera_thread.start()
        print("摄像头线程已启动，等待初始化结果...")
        
        # 初始化登录参数
        self.attempts = 0
        self.max_attempts = 3
        self.start_time = time.time()
        self.timeout = 30  # 30秒超时
    
    def start_password_login(self):
        """打开密码登录对话框"""
        # 先停止摄像头
        self.stop_camera()
        
        # 创建密码登录对话框
        login_dialog = PasswordLoginDialog(self.user_manager, self)
        result = login_dialog.exec_()
        
        if result == QDialog.Accepted:
            # 登录成功
            user_id = login_dialog.user_id
            username = login_dialog.username
            is_admin = login_dialog.is_admin
            
            # 发送登录成功信号，包含管理员标志
            self.login_successful.emit(user_id, username, is_admin)
        else:
            # 登录取消，重置UI
            self.reset_login_ui()
    
    def on_camera_initialized(self, success):
        """摄像头初始化结果处理"""
        print(f"摄像头初始化结果: {success}")
        if success:
            print("摄像头初始化成功，创建处理定时器...")
            # 创建处理定时器
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(100)  # 10fps 用于处理，显示由摄像头线程负责
            self.status_label.setText("摄像头已启动，请面对摄像头...")
            print("处理定时器已启动，等待人脸识别...")
        else:
            print("摄像头初始化失败，停止摄像头并重置UI...")
            self.stop_camera()
            self.reset_login_ui()
    
    def on_camera_error(self, error_msg):
        """处理摄像头错误"""
        self.video_frame.setText(error_msg)
        QMessageBox.critical(self, "摄像头错误", error_msg)
        self.stop_camera()
        self.reset_login_ui()
    
    def on_frame_ready(self, frame):
        """接收摄像头线程传来的帧"""
        try:
            if frame is not None:
                # 保存当前帧供处理使用
                self.current_frame = frame.copy()
                # 显示帧
                self.video_frame.display_frame(frame)
        except Exception as e:
            print(f"处理帧出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_frame(self):
        """处理视频帧并进行人脸识别"""
        try:
            # 检查是否有帧可用
            if self.current_frame is None:
                return
                
            # 使用当前帧进行处理
            frame = self.current_frame.copy()
            
            # 创建预览帧
            preview_frame = frame.copy()
            
            # 检测人脸
            faces = []
            try:
                _, detected_faces = self.user_manager.face_manager.detect_faces(frame, draw_result=False)
                
                # 确保检测的结果是有效的
                if detected_faces is not None and len(detected_faces) > 0:
                    print(f"检测到 {len(detected_faces)} 个人脸")
                    faces = detected_faces
                    
                    # 在预览帧上绘制人脸框
                    for face in faces:
                        if len(face) >= 4:  # 确保有足够的坐标
                            x1, y1, x2, y2 = [int(p) for p in face[:4]]
                            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print("未检测到人脸")
                    preview_frame = put_chinese_text(preview_frame, "请正视摄像头", (20, 40), color=(0, 255, 255), size=24)
            except Exception as e:
                # 发生错误时记录错误，但继续显示原始帧
                print(f"人脸检测错误: {str(e)}")
                import traceback
                traceback.print_exc()
                # 在预览帧上显示错误信息
                preview_frame = put_chinese_text(preview_frame, "检测错误", (20, 40), color=(0, 0, 255), size=24)
            
            # 显示状态信息
            status_text = f"请面对摄像头 (尝试 {self.attempts+1}/{self.max_attempts})"
            preview_frame = put_chinese_text(preview_frame, status_text, (20, 40), color=(0, 255, 0), size=24)
            
            # 更新状态并检查是否超时
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, int(self.timeout - elapsed_time))
            self.status_label.setText(f"{status_text} - 剩余时间: {remaining_time}秒")
            
            # 如果超时，中止登录过程
            if elapsed_time > self.timeout:
                self.status_label.setText("登录超时，请重试")
                self.video_frame.display_frame(preview_frame)  # 显示最后一帧
                self.stop_camera()
                self.reset_login_ui()
                return
            
            # 如果检测到人脸
            if len(faces) > 0:
                # 只处理检测到的第一个人脸
                face = faces[0]
                x1, y1, x2, y2 = [int(p) for p in face[:4]]
                
                # 检查人脸坐标是否在有效范围内
                if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    # 人脸坐标超出图像范围，调整坐标
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)
                
                # 在预览中绘制人脸框
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                try:
                    # 提取人脸特征
                    face_encoding = self.user_manager.face_manager.extract_face_encoding(frame, [x1, y1, x2, y2])
                    
                    if face_encoding is not None:
                        # 与数据库中的人脸比对
                        known_encodings = self.user_manager.db_manager.get_all_face_encodings()
                        if not known_encodings:
                            self.status_label.setText("数据库中没有注册的人脸，请先注册")
                            self.video_frame.display_frame(preview_frame)
                            time.sleep(2)  # 显示提示信息2秒
                            self.stop_camera()
                            self.reset_login_ui()
                            return
                            
                        user_id, username, distance = self.user_manager.face_manager.compare_faces(
                            known_encodings, face_encoding
                        )
                        
                        if user_id is not None:
                            # 登录成功
                            try:
                                # 获取用户信息，检查是否是管理员
                                user_info = self.user_manager.get_user_info(user_id)
                                is_admin = False
                                if user_info and "is_admin" in user_info:
                                    is_admin = bool(user_info["is_admin"])
                                
                                self.user_manager.db_manager.record_login(user_id, 'face')
                            except Exception as e:
                                print(f"记录登录信息失败: {str(e)}")
                                # 继续处理，不影响用户登录体验
                                is_admin = False
                            
                            # 在界面中显示成功信息
                            success_text = f"欢迎回来, {username}!"
                            preview_frame = put_chinese_text(preview_frame, success_text, (x1, y1 - 30), color=(0, 255, 0), size=24)
                            
                            # 确保显示最后一帧
                            self.video_frame.display_frame(preview_frame)
                            
                            # 延迟一下，让用户看到成功信息
                            QApplication.processEvents()
                            time.sleep(1)
                            
                            # 停止摄像头和计时器
                            self.stop_camera()
                            
                            # 发送登录成功信号，包含管理员标志
                            self.login_successful.emit(user_id, username, is_admin)
                            return
                        else:
                            # 识别失败
                            fail_text = f"未识别, 相似度: {1-distance:.2f}"
                            preview_frame = put_chinese_text(preview_frame, fail_text, (x1, y1 - 30), color=(0, 0, 255), size=24)
                            
                            self.attempts += 1
                            
                            if self.attempts >= self.max_attempts:
                                self.status_label.setText("登录失败，已达到最大尝试次数")
                                # 确保显示最后一帧
                                self.video_frame.display_frame(preview_frame)
                                # 延迟一下，让用户看到失败信息
                                QApplication.processEvents()
                                time.sleep(1)
                                self.stop_camera()
                                self.reset_login_ui()
                                return
                    else:
                        # 无法提取人脸特征
                        preview_frame = put_chinese_text(preview_frame, "无法提取特征", (x1, y1 - 30), color=(0, 0, 255), size=24)
                except Exception as e:
                    # 人脸识别过程出错
                    error_text = "识别出错"
                    preview_frame = put_chinese_text(preview_frame, error_text, (x1, y1 - 30), color=(0, 0, 255), size=24)
                    self.status_label.setText(f"人脸识别错误: {str(e)}")
                    print(f"人脸识别错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # 更新视频预览
            self.video_frame.display_frame(preview_frame)
            
        except Exception as e:
            # 整个过程出错
            print(f"处理帧时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 继续执行，避免崩溃
            self.status_label.setText("处理人脸时出错，请重试")
            self.reset_login_ui()
    
    def stop_camera(self):
        """停止摄像头和计时器"""
        print("LoginWidget: 开始停止摄像头...")
        
        if hasattr(self, 'timer') and self.timer and self.timer.isActive():
            print("LoginWidget: 停止定时器...")
            self.timer.stop()
            print("LoginWidget: 定时器已停止")
        
        # 停止摄像头线程
        if hasattr(self, 'camera_thread') and self.camera_thread:
            print("LoginWidget: 停止摄像头线程...")
            self.camera_thread.stop()
            self.camera_thread = None
            print("LoginWidget: 摄像头线程已停止")
            
        # 原有摄像头释放逻辑保留，以防万一
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            print("LoginWidget: 释放直接摄像头...")
            self.cap.release()
            if self.cap in active_cameras:
                active_cameras.remove(self.cap)
                print("LoginWidget: 摄像头已从全局列表中移除")
            self.cap = None
            print("LoginWidget: 摄像头已释放")
        
        # 清除当前帧
        self.current_frame = None
        print("LoginWidget: 摄像头停止完成")
    
    def reset_login_ui(self):
        """重置登录界面状态"""
        self.face_login_button.setText("人脸登录")
        self.face_login_button.setEnabled(True)
    
    def open_register_dialog(self):
        """打开用户注册对话框"""
        # 先停止当前摄像头
        self.stop_camera()
        
        # 打开注册对话框
        register_dialog = RegisterDialog(self.user_manager, self)
        result = register_dialog.exec_()
        
        # 无论成功与否，确保注册对话框的资源被释放
        if hasattr(register_dialog, 'cap') and register_dialog.cap and register_dialog.cap.isOpened():
            register_dialog.stop_camera()
            
        if result == QDialog.Accepted:
            # 注册成功后显示成功消息
            QMessageBox.information(self, "注册成功", "用户注册成功，现在您可以登录了")
        
        # 重置登录界面
        self.reset_login_ui()
    
    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        print("关闭LoginWidget，释放资源...")
        self.stop_camera()
        print("LoginWidget资源释放完成")
        event.accept()


class UserDashboard(QWidget):
    """用户仪表板界面"""
    logout_requested = pyqtSignal()
    
    def __init__(self, user_manager, user_id, username, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user_id = user_id
        self.username = username
        
        self.init_ui()
        self.load_user_data()
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 欢迎标题
        self.welcome_label = QLabel(f"欢迎回来，{self.username}")
        self.welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(self.welcome_label)
        
        # 登录信息
        self.login_info_label = QLabel("")
        main_layout.addWidget(self.login_info_label)
        
        # 选项卡控件
        tabs = QTabWidget()
        
        # 用户信息选项卡
        info_tab = QWidget()
        info_layout = QVBoxLayout()
        
        # 用户基本信息
        info_box = QGroupBox("用户信息")
        info_form = QGridLayout()
        
        info_form.addWidget(QLabel("用户名:"), 0, 0)
        info_form.addWidget(QLabel(self.username), 0, 1)
        
        info_form.addWidget(QLabel("用户ID:"), 1, 0)
        info_form.addWidget(QLabel(str(self.user_id)), 1, 1)
        
        info_form.addWidget(QLabel("注册时间:"), 2, 0)
        self.register_time_label = QLabel("")
        info_form.addWidget(self.register_time_label, 2, 1)
        
        info_box.setLayout(info_form)
        info_layout.addWidget(info_box)
        
        # 登录历史
        history_box = QGroupBox("登录历史")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        
        history_box.setLayout(history_layout)
        info_layout.addWidget(history_box)
        
        info_tab.setLayout(info_layout)
        tabs.addTab(info_tab, "用户信息")
        
        # 添加选项卡到主布局
        main_layout.addWidget(tabs)
        
        # 退出按钮
        self.logout_button = QPushButton("退出登录")
        self.logout_button.clicked.connect(self.logout)
        main_layout.addWidget(self.logout_button)
        
        self.setLayout(main_layout)
    
    def load_user_data(self):
        """加载用户数据"""
        # 获取用户信息
        user_info = self.user_manager.get_user_info(self.user_id)
        if user_info:
            # 格式化注册时间
            try:
                # 尝试解析日期格式
                register_time = datetime.strptime(user_info["register_time"], "%Y-%m-%d %H:%M:%S")
                self.register_time_label.setText(register_time.strftime("%Y年%m月%d日 %H:%M"))
            except (ValueError, TypeError):
                # 如果解析失败，显示一个友好的提示
                self.register_time_label.setText("未知日期")
            
            # 设置登录信息
            self.login_info_label.setText(f"当前登录时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        
        # 获取登录历史
        login_history = self.user_manager.get_user_login_history(self.user_id)
        self.history_list.clear()
        
        for login_record in login_history:
            login_time = login_record[0]
            login_method = login_record[1] if len(login_record) > 1 else "未知"
            
            try:
                time_obj = datetime.strptime(login_time, "%Y-%m-%d %H:%M:%S")
                formatted_time = time_obj.strftime("%Y年%m月%d日 %H:%M:%S")
            except ValueError:
                # 如果解析失败，直接使用原始值
                formatted_time = str(login_time)
            
            if login_method == "face":
                method_text = "人脸识别"
            elif login_method == "password":
                method_text = "密码登录"
            else:
                method_text = login_method
                
            self.history_list.addItem(f"{formatted_time} ({method_text})")
    
    def logout(self):
        """退出登录"""
        self.logout_requested.emit()


class AdminDashboard(QWidget):
    """管理员控制面板界面"""
    logout_requested = pyqtSignal()
    
    def __init__(self, user_manager, user_id, username, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.user_id = user_id
        self.username = username
        
        self.init_ui()
        self.load_user_data()
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 欢迎标题
        self.welcome_label = QLabel(f"管理员控制面板 - {self.username}")
        self.welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(self.welcome_label)
        
        # 状态信息
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: blue;")
        main_layout.addWidget(self.status_label)
        
        # 选项卡控件
        tabs = QTabWidget()
        
        # 用户管理选项卡
        user_tab = self.create_user_tab()
        tabs.addTab(user_tab, "用户管理")
        
        # 系统信息选项卡
        system_tab = self.create_system_tab()
        tabs.addTab(system_tab, "系统信息")
        
        # 添加选项卡到主布局
        main_layout.addWidget(tabs)
        
        # 退出按钮
        self.logout_button = QPushButton("退出登录")
        self.logout_button.setStyleSheet("background-color: #f44336;")
        self.logout_button.clicked.connect(self.logout)
        main_layout.addWidget(self.logout_button)
        
        self.setLayout(main_layout)
    
    def create_user_tab(self):
        """创建用户管理选项卡"""
        user_tab = QWidget()
        layout = QVBoxLayout()
        
        # 用户列表
        list_group = QGroupBox("用户列表")
        list_layout = QVBoxLayout()
        
        # 用户表格
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(6)
        self.user_table.setHorizontalHeaderLabels(["ID", "用户名", "管理员", "注册时间", "人脸数据", "操作"])
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 不可编辑
        self.user_table.setSelectionBehavior(QTableWidget.SelectRows)  # 选择整行
        self.user_table.horizontalHeader().setStretchLastSection(True)
        
        list_layout.addWidget(self.user_table)
        
        # 刷新按钮
        refresh_button = QPushButton("刷新用户列表")
        refresh_button.clicked.connect(self.load_user_data)
        list_layout.addWidget(refresh_button)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 新建用户区域
        create_group = QGroupBox("添加新用户")
        create_layout = QGridLayout()
        
        username_label = QLabel("用户名:")
        self.new_username = QLineEdit()
        create_layout.addWidget(username_label, 0, 0)
        create_layout.addWidget(self.new_username, 0, 1)
        
        password_label = QLabel("密码:")
        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        create_layout.addWidget(password_label, 1, 0)
        create_layout.addWidget(self.new_password, 1, 1)
        
        admin_label = QLabel("是否管理员:")
        self.new_is_admin = QCheckBox()
        create_layout.addWidget(admin_label, 2, 0)
        create_layout.addWidget(self.new_is_admin, 2, 1)
        
        add_button = QPushButton("添加用户")
        add_button.clicked.connect(self.add_user)
        create_layout.addWidget(add_button, 3, 0, 1, 2)
        
        create_group.setLayout(create_layout)
        layout.addWidget(create_group)
        
        user_tab.setLayout(layout)
        return user_tab
    
    def create_system_tab(self):
        """创建系统信息选项卡"""
        system_tab = QWidget()
        layout = QVBoxLayout()
        
        # 系统信息
        info_group = QGroupBox("系统信息")
        info_layout = QGridLayout()
        
        info_layout.addWidget(QLabel("数据库路径:"), 0, 0)
        info_layout.addWidget(QLabel(self.user_manager.db_manager.db_path), 0, 1)
        
        info_layout.addWidget(QLabel("人脸样本目录:"), 1, 0)
        info_layout.addWidget(QLabel("face_samples/"), 1, 1)
        
        info_layout.addWidget(QLabel("识别阈值:"), 2, 0)
        threshold_value = str(self.user_manager.face_manager.recognition_threshold)
        info_layout.addWidget(QLabel(threshold_value), 2, 1)
        
        # 添加数据库大小标签
        info_layout.addWidget(QLabel("数据库大小:"), 3, 0)
        self.db_size_label = QLabel("--")
        info_layout.addWidget(self.db_size_label, 3, 1)
        
        # 添加数据库时间标签
        info_layout.addWidget(QLabel("最后修改时间:"), 4, 0)
        self.db_time_label = QLabel("--")
        info_layout.addWidget(self.db_time_label, 4, 1)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 系统操作
        actions_group = QGroupBox("系统操作")
        actions_layout = QVBoxLayout()
        
        backup_button = QPushButton("备份数据库")
        backup_button.clicked.connect(self.backup_database)
        actions_layout.addWidget(backup_button)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        system_tab.setLayout(layout)
        return system_tab
    
    def load_user_data(self):
        """加载管理员控制面板的用户数据"""
        # 清空用户表格
        self.user_table.setRowCount(0)
        
        # 获取所有用户信息
        users = self.user_manager.get_all_users()
        if not users:
            return
        
        # 填充用户表格
        for row, user in enumerate(users):
            user_id = user[0]
            username = user[1]
            is_admin = user[2]
            register_time = user[3]
            
            self.user_table.insertRow(row)
            
            # 设置ID
            id_item = QTableWidgetItem(str(user_id))
            self.user_table.setItem(row, 0, id_item)
            
            # 设置用户名
            username_item = QTableWidgetItem(username)
            self.user_table.setItem(row, 1, username_item)
            
            # 设置管理员状态
            admin_item = QTableWidgetItem("是" if is_admin else "否")
            self.user_table.setItem(row, 2, admin_item)
            
            # 设置注册时间
            try:
                # 尝试解析日期格式
                time_obj = datetime.strptime(register_time, "%Y-%m-%d %H:%M:%S")
                time_item = QTableWidgetItem(time_obj.strftime("%Y-%m-%d %H:%M"))
            except (ValueError, TypeError):
                # 处理无效日期格式或NoneType
                time_item = QTableWidgetItem("未知日期")
            self.user_table.setItem(row, 3, time_item)
            
            # 检查用户是否有人脸数据
            has_face = self.user_manager.has_face_data(user_id)
            face_item = QTableWidgetItem("有" if has_face else "无")
            self.user_table.setItem(row, 4, face_item)
            
            # 添加操作按钮
            btn_widget = QWidget()
            btn_layout = QHBoxLayout()
            btn_layout.setContentsMargins(0, 0, 0, 0)
            
            # 删除按钮
            del_btn = QPushButton("删除")
            del_btn.setFixedWidth(40)
            del_btn.clicked.connect(lambda checked, uid=user_id: self.delete_user(uid))
            
            # 重置人脸按钮
            reset_btn = QPushButton("重置人脸")
            reset_btn.setFixedWidth(70)
            reset_btn.clicked.connect(lambda checked, uid=user_id: self.reset_user_face(uid))
            
            btn_layout.addWidget(del_btn)
            btn_layout.addWidget(reset_btn)
            btn_widget.setLayout(btn_layout)
            
            self.user_table.setCellWidget(row, 5, btn_widget)
        
        # 更新状态
        self.status_label.setText(f"共加载 {len(users)} 位用户")
        
        # 更新系统信息
        db_path = self.user_manager.db_manager.db_path
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path) / 1024.0  # KB
            self.db_size_label.setText(f"{db_size:.1f} KB")
            
            db_time = datetime.fromtimestamp(os.path.getmtime(db_path))
            self.db_time_label.setText(db_time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.db_size_label.setText("文件不存在")
            self.db_time_label.setText("--")
    
    def add_user(self):
        """添加新用户"""
        username = self.new_username.text().strip()
        password = self.new_password.text()
        is_admin = self.new_is_admin.isChecked()
        
        if not username:
            QMessageBox.warning(self, "输入错误", "请输入用户名")
            return
            
        if not password:
            QMessageBox.warning(self, "输入错误", "请输入密码")
            return
        
        # 添加用户
        user_id = self.user_manager.register_user(username, password, is_admin)
        
        if user_id:
            QMessageBox.information(self, "添加成功", f"用户 '{username}' 已成功添加")
            
            # 清空输入框
            self.new_username.clear()
            self.new_password.clear()
            self.new_is_admin.setChecked(False)
            
            # 刷新用户列表
            self.load_user_data()
        else:
            QMessageBox.critical(self, "添加失败", "无法添加用户")
    
    def delete_user(self, user_id):
        """删除用户"""
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认删除", 
            "确实要删除此用户吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.user_manager.delete_user(user_id)
            
            if success:
                self.status_label.setText(f"已删除用户 ID: {user_id}")
                self.load_user_data()
            else:
                QMessageBox.critical(self, "删除失败", "无法删除用户")
    
    def reset_user_face(self, user_id):
        """重置用户的人脸数据"""
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认重置", 
            "确实要重置此用户的人脸数据吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.user_manager.reset_user_face(user_id)
            
            if success:
                self.status_label.setText(f"已重置用户 ID: {user_id} 的人脸数据")
                self.load_user_data()
            else:
                QMessageBox.critical(self, "重置失败", "无法重置用户的人脸数据")
    
    def backup_database(self):
        """备份数据库"""
        import shutil
        import datetime
        
        try:
            # 生成备份文件名
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(BACKUP_DIR, f"face_system_{now}.db")
            
            # 复制数据库文件
            shutil.copy2(self.user_manager.db_manager.db_path, backup_file)
            
            self.status_label.setText(f"数据库已备份到: {backup_file}")
            QMessageBox.information(self, "备份成功", f"数据库已备份到: {backup_file}")
        except Exception as e:
            QMessageBox.critical(self, "备份失败", f"无法备份数据库: {str(e)}")
    
    def logout(self):
        """退出登录"""
        self.logout_requested.emit()


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        print("初始化MainWindow...")
        self.user_manager = UserManager()
        print("初始化UserManager成功")
        
        self.setWindowTitle("人脸识别登录系统")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QLabel { font-size: 14px; }
            QPushButton { 
                height: 35px; 
                border-radius: 4px; 
                background-color: #2196F3; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 5px 15px;
            }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:disabled { background-color: #cccccc; }
            QGroupBox { 
                font-size: 14px; 
                font-weight: bold; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                margin-top: 10px; 
                padding: 10px;
            }
            QTabWidget::pane { 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 5px;
            }
            QTabBar::tab { 
                background-color: #e1e1e1; 
                padding: 8px 12px; 
                margin-right: 2px; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { 
                background-color: #fff; 
                border-bottom: 2px solid #2196F3;
            }
        """)
        
        self.init_ui()
        print("MainWindow初始化完成")
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中心部件
        center_widget = QWidget()
        self.setCentralWidget(center_widget)
        
        # 创建布局
        self.main_layout = QVBoxLayout(center_widget)
        
        # 创建登录界面
        self.login_widget = LoginWidget(self.user_manager)
        self.login_widget.login_successful.connect(self.on_login_successful)
        
        # 添加登录界面
        self.main_layout.addWidget(self.login_widget)
        
        # 初始时显示登录界面
        self.dashboard_widget = None
        self.admin_dashboard_widget = None
    
    def on_login_successful(self, user_id, username, is_admin):
        """处理登录成功事件"""
        try:
            # 移除登录界面
            if self.login_widget:
                # 确保先停止摄像头
                self.login_widget.stop_camera()
                self.login_widget.hide()
                self.main_layout.removeWidget(self.login_widget)
            
            if is_admin:
                # 创建并显示管理员仪表板
                self.admin_dashboard_widget = AdminDashboard(self.user_manager, user_id, username)
                self.admin_dashboard_widget.logout_requested.connect(self.on_logout_requested)
                self.main_layout.addWidget(self.admin_dashboard_widget)
            else:
                # 创建并显示普通用户仪表板
                self.dashboard_widget = UserDashboard(self.user_manager, user_id, username)
                self.dashboard_widget.logout_requested.connect(self.on_logout_requested)
                self.main_layout.addWidget(self.dashboard_widget)
        except Exception as e:
            print(f"登录成功切换界面时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 重置界面
            if hasattr(self, 'login_widget') and self.login_widget:
                self.login_widget.reset_login_ui()
            QMessageBox.critical(self, "错误", "登录后切换界面时出错，请重试")
    
    def on_logout_requested(self):
        """处理退出登录请求"""
        # 移除仪表板界面
        if self.dashboard_widget:
            self.dashboard_widget.hide()
            self.main_layout.removeWidget(self.dashboard_widget)
            self.dashboard_widget = None
        
        if self.admin_dashboard_widget:
            self.admin_dashboard_widget.hide()
            self.main_layout.removeWidget(self.admin_dashboard_widget)
            self.admin_dashboard_widget = None
        
        # 重新显示登录界面
        self.login_widget = LoginWidget(self.user_manager)
        self.login_widget.login_successful.connect(self.on_login_successful)
        self.main_layout.addWidget(self.login_widget)
    
    def closeEvent(self, event):
        """窗口关闭事件，确保释放所有资源"""
        print("关闭主窗口，释放所有摄像头资源...")
        # 如果登录窗口存在，停止其摄像头
        if hasattr(self, 'login_widget') and self.login_widget:
            self.login_widget.stop_camera()
        
        # 确保释放所有摄像头资源
        release_all_cameras()
        event.accept()


class PasswordLoginDialog(QDialog):
    """密码登录对话框"""
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("密码登录")
        self.setMinimumSize(400, 200)
        self.setStyleSheet("""
            QDialog { background-color: #f5f5f5; }
            QLabel { font-size: 14px; }
            QLineEdit { 
                height: 30px; 
                border-radius: 4px; 
                border: 1px solid #ccc; 
                padding: 5px; 
                font-size: 14px; 
            }
            QPushButton { 
                height: 35px; 
                border-radius: 4px; 
                background-color: #2196F3; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 表单布局
        form_layout = QGridLayout()
        
        # 用户名输入
        username_label = QLabel("用户名:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addWidget(username_label, 0, 0)
        form_layout.addWidget(self.username_input, 0, 1)
        
        # 密码输入
        password_label = QLabel("密码:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addWidget(password_label, 1, 0)
        form_layout.addWidget(self.password_input, 1, 1)
        
        layout.addLayout(form_layout)
        
        # 状态消息
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.login)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("background-color: #f44336;")
        
        buttons_layout.addWidget(self.login_button)
        buttons_layout.addWidget(self.cancel_button)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def login(self):
        """执行登录操作"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username:
            self.status_label.setText("请输入用户名")
            return
        
        if not password:
            self.status_label.setText("请输入密码")
            return
        
        # 提示用户正在登录
        self.status_label.setText("正在验证...")
        self.status_label.setStyleSheet("color: blue;")
        self.login_button.setEnabled(False)
        QApplication.processEvents()  # 确保UI更新
        
        # 验证用户名和密码
        user = self.user_manager.login_with_password(username, password)
        
        if user:
            self.user_id = user["user_id"]
            self.username = user["username"]
            self.is_admin = user["is_admin"]
            self.accept()
        else:
            self.status_label.setText("用户名或密码错误")
            self.status_label.setStyleSheet("color: red;")
            self.login_button.setEnabled(True)
            # 清空密码输入框
            self.password_input.setText("")


def main():
    print("=== 程序启动 ===")
    print("Python版本:", sys.version)
    print("OpenCV版本:", cv2.__version__)
    print("初始化全局变量...")
    print(f"DO_FLIP={DO_FLIP}, FLIP_MODE={FLIP_MODE}")
    print(f"LOW_PERFORMANCE={LOW_PERFORMANCE}, MAX_PROCESS_WIDTH={MAX_PROCESS_WIDTH}")
    
    try:
        print("创建QApplication...")
        app = QApplication(sys.argv)
        print("创建MainWindow...")
        window = MainWindow()
        print("显示MainWindow...")
        window.show()
        print("进入事件循环...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 