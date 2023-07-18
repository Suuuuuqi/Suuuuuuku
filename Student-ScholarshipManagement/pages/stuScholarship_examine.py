# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:47:09 2023

@author: wp
"""

from enum import auto
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
col1,col2=st.columns([1,4])

col2.markdown("# 奖学金申请流程审批 🧾")
st.sidebar.markdown("# 奖学金申请流程审批")
#操作按钮
def Examine():
    c41,c42,c43,c44,c45,c46,c47,c48=st.columns(8)
    c44.button('拒绝')
    c46.button('通过')

#选择操作
choice1 = st.sidebar.selectbox('   请下拉选择框', ["待办事项","审批详情","审批材料"])
if choice1=='待办事项':
    #显示待办事项
    image=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a1.png')
    st.image(image)
elif choice1 == '审批详情':
     #显示个人记录
    image=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a2.png')
    st.image(image)
elif choice1 == '审批材料':
     #显示个人记录
    image1=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.png')
    st.image(image1)
    #如果勾选就显示数据框
    c51,c52,c53,c54,c55=st.columns(5)
    if c51.checkbox('学习成绩'):
        image2=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.1.png')
        st.image(image2)
        Examine()
    if c52.checkbox('科研成果'):
        image2=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.2.png')
        st.image(image2)
        Examine()
    if c53.checkbox('科创竞赛'):
        image2=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.3.png')
        st.image(image2)
        Examine()
    if c54.checkbox('民主评议'):
        image2=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.4.png')
        st.image(image2)
        Examine()
    if c55.checkbox('社会活动'):
        image2=Image.open(r'D:\Desktop\Student-ScholarshipManagement\pictures\a3.5.png')
        st.image(image2)
        Examine()









