import os
import glob

# 需要处理的目录
dirs = [
    "/media/lht/disk/code/UltraFusion/results/LowLight",
    "/media/lht/disk/code/UltraFusion/results-tiled/LowLight",
]

for d in dirs:
    if not os.path.isdir(d):
        print(f"[跳过] 目录不存在: {d}")
        continue

    files = glob.glob(os.path.join(d, "*_out.png"))
    print(f"\n[{d}] 找到 {len(files)} 个文件")
    count = 0
    for old_path in files:
        new_path = old_path.replace("_out.png", ".png")
        os.rename(old_path, new_path)
        print(f"  {os.path.basename(old_path)}  ->  {os.path.basename(new_path)}")
        count += 1
    print(f"完成，重命名了 {count} 个文件")

print("\n全部完成!")
