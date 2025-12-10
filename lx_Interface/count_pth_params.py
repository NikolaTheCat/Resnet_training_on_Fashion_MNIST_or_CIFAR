import os
import pickle
import glob
import argparse
import struct
import random

def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def estimate_parameters(file_size, data_type='float32'):
    """通过文件大小估算参数量"""
    # 不同数据类型的字节数
    byte_size = {
        'float16': 2,
        'float32': 4,
        'float64': 8,
        'int8': 1,
        'int16': 2,
        'int32': 4,
        'int64': 8
    }
    # 使用float32作为默认估计
    return file_size / byte_size.get(data_type, 4)

def find_model_files(directory, recursive=False, extensions=['.pth', '.pt']):
    """查找模型文件"""
    if recursive:
        model_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    model_files.append(os.path.join(root, file))
        return model_files
    else:
        model_files = []
        for ext in extensions:
            model_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        return model_files

def create_sample_model():
    """创建一个示例模型文件用于测试"""
    import random
    import os
    
    # 创建不同大小的示例文件
    sample_sizes = {
        "small_model.pth": 1024 * 1024,  # 1MB
        "medium_model.pth": 5 * 1024 * 1024,  # 5MB
        "large_model.pth": 20 * 1024 * 1024  # 20MB
    }
    
    for filename, size in sample_sizes.items():
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        # 创建包含模型关键字的二进制数据
        with open(file_path, 'wb') as f:
            # 写入一些头信息模拟PyTorch模型文件
            f.write(b"PK\x03\x04")  # ZIP文件头的开始
            f.write(b"state_dict_sample_model")  # 模拟state_dict关键字
            # 填充随机数据到指定大小
            remaining_size = size - f.tell()
            if remaining_size > 0:
                f.write(os.urandom(remaining_size))
        print(f"创建示例模型文件: {filename} ({format_number(size)})")
    
    return [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in sample_sizes.keys()]

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='统计PyTorch模型文件的参数信息')
    parser.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)),
                       help='要搜索的目录路径（默认为当前目录）')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='是否递归搜索子目录')
    parser.add_argument('--create-sample', action='store_true',
                       help='创建示例模型文件用于测试')
    
    args = parser.parse_args()
    
    # 创建示例模型文件
    if args.create_sample:
        print("正在创建示例模型文件...")
        sample_files = create_sample_model()
        print("示例文件创建完成！")
        print("\n现在可以运行以下命令来测试：")
        print("  python count_pth_params.py")
        return
    
    directory = args.dir
    
    # 验证目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在！")
        return
    
    # 查找模型文件
    model_files = find_model_files(directory, args.recursive)
    
    if not model_files:
        search_mode = "递归搜索" if args.recursive else "当前目录"
        print(f"在 {search_mode} '{directory}' 中未找到.pth或.pt模型文件")
        print("\n使用说明：")
        print("  1. 搜索当前目录：python count_pth_params.py")
        print("  2. 递归搜索：python count_pth_params.py --recursive")
        print("  3. 指定目录：python count_pth_params.py --dir 你的目录路径")
        print("  4. 创建示例文件：python count_pth_params.py --create-sample")
        return
    
    print(f"找到 {len(model_files)} 个模型文件，开始分析...")
    print("-" * 120)
    print(f"{'文件名':<30} {'相对路径':<50} {'文件大小':<15} {'估计参数量':<15}")
    print("-" * 120)
    
    # 按文件大小排序
    model_files.sort(key=os.path.getsize, reverse=True)
    
    total_size = 0
    total_params = 0
    
    for model_file in model_files:
        file_name = os.path.basename(model_file)
        # 计算相对路径
        rel_path = os.path.relpath(model_file, directory)
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(model_file)
            total_size += file_size
            
            # 估算参数量（假设float32精度）
            estimated_params = estimate_parameters(file_size)
            total_params += estimated_params
            
            # 格式化输出
            size_str = format_number(file_size)
            params_str = format_number(estimated_params)
            print(f"{file_name:<30} {rel_path:<50} {size_str:<15} {params_str:<15}")
                
        except Exception as e:
            print(f"{file_name:<30} {rel_path:<50} {'错误':<15} {str(e)[:30]}...")
    
    print("-" * 120)
    print(f"总计：{len(model_files)} 个文件，总大小：{format_number(total_size)}，估计总参数量：{format_number(total_params)}")
    print("\n注意：")
    print("  1. 参数量为基于文件大小的估算值（假设float32精度，每个参数4字节）")
    print("  2. 实际参数量可能因数据类型和模型结构而有所不同")
    print("  3. 如需准确计算，请确保PyTorch环境正确安装，并使用torch.load加载模型")
    print("\n使用示例：")
    print("  - 创建测试示例：python count_pth_params.py --create-sample")
    print("  - 递归搜索整个项目：python count_pth_params.py --dir .. --recursive")
    print("\n准确计算参数量的PyTorch代码示例：")
    print("  import torch")
    print("  model = torch.load('model.pth', map_location='cpu')")
    print("  if isinstance(model, dict) and 'state_dict' in model:")
    print("      params = sum(p.numel() for p in model['state_dict'].values())")
    print("  else:")
    print("      params = sum(p.numel() for p in model.parameters())")
    print("  print(f'模型参数量: {params:,}')")


if __name__ == "__main__":
    main()