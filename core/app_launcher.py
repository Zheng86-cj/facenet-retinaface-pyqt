#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人脸识别登录系统主程序入口
"""

import os
import sys
import time
import argparse

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(current_dir)

# 添加调试日志
import logging
import traceback
from datetime import datetime

# 配置日志
logs_dir = os.path.join(root_dir, 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

debug_log_file = os.path.join(logs_dir, f'face_login_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(debug_log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

debug_logger = logging.getLogger('FaceLogin')
debug_logger.info("=== 程序启动 ===")

# 设置全局异常处理
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    debug_logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
    print("程序发生错误，详细信息已记录到日志:", debug_log_file)
    
sys.excepthook = handle_exception

from PyQt5.QtWidgets import QApplication
from core.gui_controller import MainWindow

def check_directories():
    """确保必要的目录存在"""
    dirs_to_check = [os.path.join(root_dir, "face_samples")]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            print(f"创建目录: {directory}")
            os.makedirs(directory)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="人脸识别登录系统")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认为0")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU，即使CUDA可用")
    parser.add_argument("--low-performance", action="store_true", help="低性能模式，减少处理分辨率以提高速度")
    parser.add_argument("--no-flip", action="store_true", help="不进行图像翻转")
    parser.add_argument("--flip-mode", type=int, default=-1, choices=[0, 1, -1], help="图像翻转模式: 0=上下翻转, 1=左右翻转, -1=上下左右翻转")
    return parser.parse_args()

def main():
    """主函数"""
    try:
        print("人脸识别登录系统启动...")
        args = parse_args()
        print(f"命令行参数: {args}")
        
        # 检查目录
        check_directories()
        
        # 如果强制使用CPU，设置环境变量
        if args.force_cpu:
            print("强制使用CPU模式")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # 创建并启动应用
        print("初始化QApplication...")
        app = QApplication(sys.argv)
        
        print("创建MainWindow...")
        window = MainWindow()
        
        # 设置摄像头索引
        if hasattr(window.login_widget, 'camera_index'):
            print(f"设置摄像头索引: {args.camera}")
            window.login_widget.camera_index = args.camera
        else:
            print("警告: 无法设置摄像头索引，login_widget没有camera_index属性")
        
        # 设置图像翻转模式
        print("导入gui_controller模块...")
        from core import gui_controller
        if args.no_flip:
            # 全局变量记录是否翻转
            print("禁用图像翻转")
            gui_controller.DO_FLIP = False
        else:
            print(f"启用图像翻转，模式: {args.flip_mode}")
            gui_controller.DO_FLIP = True
            gui_controller.FLIP_MODE = args.flip_mode
        
        # 设置低性能模式
        if args.low_performance:
            print("启用低性能模式")
            # 配置全局变量以减少处理分辨率
            gui_controller.LOW_PERFORMANCE = True
            gui_controller.MAX_PROCESS_WIDTH = 320
        else:
            gui_controller.LOW_PERFORMANCE = False
            gui_controller.MAX_PROCESS_WIDTH = 640
        
        # 显示窗口
        print("显示主窗口...")
        window.show()
        
        # 启动应用事件循环
        print("进入事件循环...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 