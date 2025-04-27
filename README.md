# 人脸识别系统

## 核心功能
- 实时人脸检测与识别
- 用户注册/登录管理
- 多模型支持（MobileNet/Inception-ResNet）
- 可视化操作界面

## 技术架构
```
├── core/              # 核心业务逻辑
│   ├── gui_controller.py   # 图形界面控制
│   └── user_service.py      # 用户管理服务
├── models/            # 模型实现
│   ├── face_encoder.py      # 人脸编码器
│   └── face_recognition.py  # 识别算法实现
├── data/              # 样本数据存储
└── app.py             # 应用入口
```

## 快速启动
```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

## 配置参数
```python
# app.py 示例配置
DETECTION_THRESHOLD = 0.7   # 人脸检测置信度
RECOGNITION_THRESHOLD = 0.4 # 识别相似度阈值
CAMERA_INDEX = 0           # 摄像头设备索引
```

## 贡献指南
请参考[CONTRIBUTING.md](CONTRIBUTING.md)中的开发规范，主要包含：
- PEP8 代码风格要求
- 单元测试编写规范
- 提交信息格式标准

## 依赖环境
```
Python 3.8+
PyQt5 5.15+  # 图形界面框架
OpenCV 4.11+  # 图像处理
Torch 2.6+    # 深度学习框架
```