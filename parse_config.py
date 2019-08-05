#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def parse_model_config(path):
    f = open(path, 'r') #读取文件
    module_defs = [] # 创建列表, 列表中的元素为字典
    for line in f.readlines(): # 逐行读取
        line = line.strip() # 消除行头尾的空白符(空格, 回车等)
        if not line or line.startswith('#'): # 如果遇到空行或者注释行, 则跳过
            continue
        if line.startswith('['):# 遇到模块的起始, 在列表后添加新的字典
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].strip() # 根据参数值为字典赋值
            if(module_defs[-1]['type']=="convolutional"):
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split('=')# 根据参数值为字典赋值, 注意要去除空白符
            module_defs[-1][key.strip()] = value.strip()
    return module_defs

