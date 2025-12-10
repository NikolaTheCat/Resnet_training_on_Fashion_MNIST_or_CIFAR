#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动 TensorBoard 查看训练日志

使用方法：
    python view_tensorboard.py
    
或指定特定日志目录：
    python view_tensorboard.py --logdir path/to/specific/log
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_log_directories():
    """查找所有可用的 TensorBoard 日志目录"""
    # 获取项目根目录的 log 文件夹
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "log"
    
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        print("请先运行训练脚本生成日志。")
        return []
    
    # 查找所有子目录
    subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"日志目录为空: {log_dir}")
        print("请先运行训练脚本生成日志。")
        return []
    
    return subdirs


def main():
    parser = argparse.ArgumentParser(description="启动 TensorBoard 查看训练日志")
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="指定 TensorBoard 日志目录路径（默认为 ../log/）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard 服务器端口（默认 6006）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="TensorBoard 服务器主机（默认 localhost）"
    )
    
    args = parser.parse_args()
    
    # 确定日志目录
    if args.logdir:
        logdir = Path(args.logdir)
    else:
        project_root = Path(__file__).parent.parent
        logdir = project_root / "log"
    
    # 检查日志目录是否存在
    if not logdir.exists():
        print(f"❌ 错误：日志目录不存在: {logdir}")
        print("\n请先运行训练脚本生成日志：")
        print("  python cifar_resnet_trainer.py --model resnet18 --cifar-version 10 ...")
        sys.exit(1)
    
    # 显示可用的日志目录
    subdirs = [d for d in logdir.iterdir() if d.is_dir()]
    if subdirs:
        print("\n" + "="*60)
        print("📊 可用的训练日志目录：")
        print("="*60)
        for i, subdir in enumerate(subdirs, 1):
            print(f"  {i}. {subdir.name}")
        print("="*60)
    
    # 启动 TensorBoard
    print(f"\n🚀 启动 TensorBoard 服务器...")
    print(f"   日志目录: {logdir}")
    print(f"   访问地址: http://{args.host}:{args.port}")
    print("\n按 Ctrl+C 停止 TensorBoard 服务器\n")
    print("="*60 + "\n")
    
    try:
        # 启动 TensorBoard
        cmd = [
            "tensorboard",
            "--logdir", str(logdir),
            "--port", str(args.port),
            "--host", args.host
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✅ TensorBoard 服务器已停止")
    except FileNotFoundError:
        print("\n❌ 错误：找不到 tensorboard 命令")
        print("\n请先安装 TensorBoard：")
        print("  pip install tensorboard")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

