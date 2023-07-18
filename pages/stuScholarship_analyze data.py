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
col1,col2=st.columns([1,2])

col2.markdown("# 学生数据分析 👩‍🎓")
st.sidebar.markdown("# 学生数据分析")

#导入数据
df=pd.read_csv(r".\data1.csv")


st.write('---')
col31,col32,col33=st.columns([1,1,1])
st.write('🗂️ 请选择数据集')
#选择数据
choice = st.selectbox('📃  请下拉选择框', ["2024届硕士生数据集","2025届硕士生数据集","2026届硕士生数据集"])


#导入表格
st.dataframe(df.iloc[0:9])
#图标
image=Image.open(r'.\pictures\5.png')
st.image(image)
st.write('---')

#四部分数据
col31,col32,col33=st.columns([1,1,1])
col32.write('第一部分：综合情况')

c41,c42,c43,c44=st.columns(4)
c41.metric("获得奖学金人数",30)
c42.metric("学习成绩平均分",80.3)
c43.metric("科研成果平均分",34.4)
c44.metric("科创竞赛平均分",60.8)
c44.metric("综合成绩平均分",62.8)
c43.metric("民主评议平均分",74.5)
c42.metric("社会活动平均分",56.7)
c41.metric("总人数",30)

#画交互图
col31,col32,col33=st.columns([1,1,1])
col32.write('第二部分：统计分析')


option = st.selectbox(
    '📃  请选择需要分析的指标',
     df.columns.tolist())

r1=df.groupby(option).size()

option1 = {
    
     
    "color":'#54BCBD',
    "tooltip": {
  "trigger": 'axis',
  "axisPointer": {
    "type": 'shadow'
  }
},
    "xAxis": {
        "type": "category",
        "data": r1.index.tolist(),
        "axisTick": {"alignWithLabel": True},
    },
    "yAxis": {"type": "value"},
    "series": [
        {"data": r1.values.tolist(), "type": "bar"}
    
    ],
    
}
st_echarts(options=option1)

#图标
image=Image.open(r'.\pictures\6.png')
st.image(image)
st.write('---')

#选择

choice1 = st.selectbox('📃  学号', ["222001","222002","222003","222004","2026届硕士生数据集"])
choice2 = st.selectbox('📃  指标', ["学习成绩","科研成果","科创竞赛","民主评议","社会活动"])
if choice2 == "学习成绩":
    #多功能柱状图
    b = (
        
        Bar()
   
        .add_xaxis(["222222001",
                    222222002,
                    222222003,
                    222222004,
                    222222005,
                    222222006,
                    222222007,
                    222222008,
                    222222009,
                    222222010,
                    222222011,
                    222222012,
                    222222013,
                    222222014,
                    222222015,
                    222222016,
                    ])
        .add_yaxis(
            "2023年的学习成绩分数", [90,
                                    89,
                                    93,
                                    80,
                                    99,
                                    80,
                                    89,
                                    94,
                                    88,
                                    85,
                                    85,
                                    80,
                                    91,
                                    90,
                                    86,
                                    82,
                                    ],
            
                                    
        )
        
        .set_global_opts(
            title_opts=opts.TitleOpts(
                 subtitle="2023年"
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
    )
    st_pyecharts(b)
    
if choice2 == "科创竞赛":
    #多功能柱状图
    c = (
        Bar()
        .add_xaxis(["国家级三等奖",
                    "省级一等奖",
                    "省级二等奖",
                    "省级三等奖",
                    "校级一等奖",
                    "校级二等奖",
                    "院级一等奖",
                    "院级二等奖",
                    "院级三等奖",

                    ])
        .add_yaxis(
            "2023年的科创比赛加分", [10,
                                    8,
                                    7,
                                    6,
                                    5,
                                    4,
                                    3,
                                    2,
                                    1,
                                    ],
           
                                 
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                 subtitle="2023年"
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
    )
    st_pyecharts(c)