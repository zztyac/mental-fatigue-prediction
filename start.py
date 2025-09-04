# -*- coding: utf-8 -*-

"""
金属多轴疲劳寿命预测系统 - 快速启动脚本
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 命令行参数
    cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # 构建启动命令
    cmd = [sys.executable, os.path.join(current_dir, "fatigue_prediction", "run.py"), "--debug"]
    cmd.extend(cmd_args)
    
    # 打印提示信息
    print("正在启动金属多轴疲劳寿命预测系统...")
    print(f"命令: {' '.join(cmd)}")
    print("\n按下 Ctrl+C 停止服务器\n")
    
    try:
        # 启动应用
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务器已停止")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1) 