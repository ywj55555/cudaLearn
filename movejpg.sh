#!/bin/bash

# 检查文件夹是否存在，如果不存在则创建
if [ ! -d "./png" ]; then
  mkdir ./png
fi

# 将当前目录下的图像移到png文件夹中
mv ./*.png ./png/
mv ./*.jpg ./png/
