import os
import glob
import sys

def rename_files(root_dir):
    """递归查找并重命名包含 _swir_original 的文件"""
    if not os.path.isdir(root_dir):
        print(f"[错误] 目录不存在: {root_dir}")
        return

    # 递归查找所有包含 _swir_original 的文件
    pattern = os.path.join(root_dir, "**", "*_swir_original*")
    files = glob.glob(pattern, recursive=True)
    print(f"\n[{root_dir}] 找到 {len(files)} 个文件")

    count = 0
    for old_path in files:
        dir_name = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        new_name = old_name.replace("_swir_original", "")
        new_path = os.path.join(dir_name, new_name)

        if old_path == new_path:
            continue

        os.rename(old_path, new_path)
        print(f"  {old_name}  ->  {new_name}")
        count += 1

    print(f"完成，重命名了 {count} 个文件")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python rename_remove_swir_original.py <数据集目录> [数据集目录2 ...]")
        print("示例: python rename_remove_swir_original.py /path/to/dataset")
        sys.exit(1)

    for d in sys.argv[1:]:
        rename_files(d)

    print("\n全部完成!")
