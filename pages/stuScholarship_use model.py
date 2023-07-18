# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:47:09 2023

@author: wp
"""

import streamlit as st 
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
import numpy as np 
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from PIL import Image

#标题
col1,col2=st.columns([1,15])

col2.markdown("# 大连理工大学奖学金评定 🗳️")
st.sidebar.markdown("# 大连理工大学奖学金评定")

df_new=pd.read_csv(r".\data2.csv")
#图标
image=Image.open(r'.\pictures\3.png')
st.image(image)



st.write('---')
st.write('✏️  第一步：请选择数据集')
#选择数据
choice = st.selectbox('📃  请下拉选择框', ["2024届硕士生数据集","2025届硕士生数据集","2026届硕士生数据集"])
#本地导入

uploaded_file = st.file_uploader("📃  选择文件")



#导入表格
st.dataframe(df_new.iloc[0:9])
df_new_copy=df_new.copy()
#数据预处理
col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
col33.write('数据预处理结果')
df_new_copy.drop(axis=1,columns=["学号"],inplace=True)

df_new_copy=pd.get_dummies(df_new_copy,columns=["性别","专业","班级"])
st.dataframe(df_new_copy.iloc[0:9])
st.write('✏️  第二步：请设定评定模型')
#选择模型
choice = st.selectbox('📃  请下拉选择框', ["[已调参] Random Forest","[已调参] SVC","[已调参] K Neighbors Classifier","[已调参] 自定义系数"])
#开始模型
col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
#col33.button("开始评定",key="abc")
if col33.button("开始评定"):
   
    
    #图标
    image=Image.open(r'.\pictures\4.png')
    st.image(image)
    st.write('---')
    #导出结果
    df_new_finished=pd.read_csv(r".\data3.csv")
    #导入表格
    st.dataframe(df_new_finished.iloc[0:9])
    #提交结果
    col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
    col33.button("提交结果")


