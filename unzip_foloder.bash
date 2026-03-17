#!/bin/bash

# 定义源目录和目标根目录
SRC_DIR="/home/lgz/dataset/ADEC/achive"
DEST_ROOT="/home/lgz/dataset/ADEC/carla/dataset"

# 确保目标根目录存在
mkdir -p "$DEST_ROOT"

for file in "$SRC_DIR"/Experiment*.zip; do
    # 1. 获取文件名（不带路径和后缀），例如 "Experiment1"
    base_name=$(basename "$file" .zip)
    
    # 2. 检查压缩包内部的第一级内容是否已经包含该目录
    # unzip -l 列表显示，awk 提取路径部分，grep 检查开头
    if unzip -l "$file" | awk 'NR==4 {print $4}' | grep -q "^${base_name}/"; then
        echo ">>> [跳过建目录] $base_name 内部已包含根目录，直接解压..."
        unzip -o "$file" -d "$DEST_ROOT"
    else
        echo ">>> [新建目录] $base_name 内部无根目录，正在创建并解压..."
        mkdir -p "$DEST_ROOT/$base_name"
        unzip -o "$file" -d "$DEST_ROOT/$base_name"
    fi
done

echo "任务完成！"