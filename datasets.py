#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# In[2]:


class ImageFolder(Dataset):
    def __init__(self,folder_path,img_size=416):
        self.files=sorted(glob.glob('%s/*.*'%folder_path))
        self.img_shape=(img_size,img_size)
    def __getitem__(self,index):
        img_path=self.files[index%len(slef.files)]
        # 获取图片
        img =np.array(Image.open(img_path))
        height,width,_=img.shape
        dim_diff=np.abs(height-width)
        pad1,pad2=dim_dif//2,dim_diff-dim_diff//2
        
        #填充矩形图像
        pad=None
        if height<= weight:
            pad=((pad1,pad2),(0,0),(0,0))
        else:
            pad=((0, 0), (pad1, pad2), (0, 0))
            
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
         
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        
        #把输入图像的通道数放到最前面
        input_img=np.transpose(input_img,(2,0,1))
        input_img=torch.from_numpy(input_img).float()
        return img_path,input_img
    
    def __len__(self):
        return len(self.files)


# In[ ]:


class ListDateset(Dataset):
    def __init__(self,list_path,img_size=416):
        with open(list_path,'r') as file:
            self.img_files=file.readlines()
            self.label_files=[
                path.replace('images','labels').replace('.png','.txt').replace('.jpg','.txt')
                for path in self.img_files
            ]
            #定义图像大小
            self.img_shape=(img_size,img_size)
            #定义图像最多能检测目标数
            slef.max_objects=50
            
    def __getitem__(self,index):
        img_path=self.img_files[index%len(self.img_files)].rstrip()
        img=np.array(Image.open(img_path))
        
        while len(img.shape)!=3:
            index=index+1
            img_path=self.img_files[index%len(self.img_files)].rstrip()
            img=np.array(Image.open(img_path))
        
        h,w,_=img.shape
        dim_diff=np.abs(h-w)
        pad1,pad2=dim_diff//2,dim_diff-dim_diff//2
        
        pad=None
        if height<= weight:
            pad=((pad1,pad2),(0,0),(0,0))
        else:
            pad=((0, 0), (pad1, pad2), (0, 0))
        
        input_img=np.pad(img,pad,'constant',constant_values=128)/255 
        
        padded_h,padded_w,_=input_img.shape
        
        input_img=resize(input_img,(*self.img_shape,3),mode='reflect')
        input_img=np.transpose(input_img,(2,0,1))
        
        input_img=torch.from_numpy(input_img).float()
        
        #label_files存储的是每一个训练样本的标签的链接
        label_path=self.label_files[index%len(self.img_files)].rstrip()
        
        labels=None
        
        
        # 50 x 5的格式,50代表最多可以识别50个物体
        # 5 代表 (class_id, x, y, w, h)
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            x1 = w * (labels[:, 1] - labels[:, 3] / 2) # 先获取box左上角和右下角的像素坐标
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)

            # 根据 padding 的大小, 更新这些坐标的值
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            # 重新将坐标转化成小数模式(相对应padding后的宽高的比例)
            labels[:, 1] = ((x1+x2)/2) / padded_w
            labels[:, 2] = ((y1+y2)/2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        filled_labels = np.zeros((self.max_objects, 5)) # 创建50×5的占位空间
        if labels is not None: 
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # 将 label 转化成 tensor
        filled_labels =torch.from_numpy(filled_labels)

        # 返回图片路径, 图片tensor, label tensor
        return img_path, input_img, filled_labels

