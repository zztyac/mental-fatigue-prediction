# -*- coding: utf-8 -*-

"""
金属多轴疲劳寿命预测系统 - 启动脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动金属多轴疲劳寿命预测系统')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口号')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    return parser.parse_args()

def setup_logging(log_level):
    """设置日志"""
    # 确保日志目录存在
    log_dir = os.path.join(project_root, 'fatigue_prediction', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'server.log'))
        ]
    )

def check_dependencies():
    """检查依赖项"""
    try:
        import flask
        import torch
        import numpy
        import pandas
        import matplotlib
        import werkzeug
    except ImportError as e:
        print(f"错误: 缺少必要的依赖项 - {e}")
        print("请运行 'pip install -r requirements.txt' 安装所有依赖")
        sys.exit(1)

def setup_directories():
    """设置必要的目录"""
    dirs = [
        os.path.join(project_root, 'uploads'),
        os.path.join(project_root, 'fatigue_prediction', 'logs'),
        os.path.join(project_root, 'fatigue_prediction', 'checkpoints'),
        os.path.join(project_root, 'dataset'),
        os.path.join(project_root, 'dataset', 'All data_Strain'),
        os.path.join(project_root, 'dataset', 'All data_Stress'),
        os.path.join(project_root, 'results'),
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def print_banner():
    """打印启动Banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███████╗ █████╗ ████████╗██╗ ██████╗ ██╗   ██╗███████╗                  ║
║   ██╔════╝██╔══██╗╚══██╔══╝██║██╔════╝ ██║   ██║██╔════╝                  ║
║   █████╗  ███████║   ██║   ██║██║  ███╗██║   ██║█████╗                    ║
║   ██╔══╝  ██╔══██║   ██║   ██║██║   ██║██║   ██║██╔══╝                    ║
║   ██║     ██║  ██║   ██║   ██║╚██████╔╝╚██████╔╝███████╗                  ║
║   ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝                  ║
║                                                                           ║
║   ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██║
║   ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
║   ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
║   ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║
║   ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
║   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
║                                                                           ║
║                                                                           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """主函数"""
    args = parse_args()
    
    # 打印Banner
    print_banner()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 检查依赖
    check_dependencies()
    
    # 设置目录
    setup_directories()
    
    # 导入应用
    try:
        from fatigue_prediction.web.app import app
    except ImportError:
        logging.error("无法导入应用，请确保web目录结构正确")
        sys.exit(1)
    
    # 提示信息
    print(f"\n服务器启动中...")
    print(f"访问地址: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"调试模式: {'开启' if args.debug else '关闭'}")
    print(f"日志级别: {args.log_level}")
    print("\n按下 Ctrl+C 停止服务器\n")
    
    # 启动应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        sys.exit(0)
    except Exception as e:
        logging.error(f"启动失败: {e}", exc_info=True)
        sys.exit(1) 