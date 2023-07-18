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



col1,col2=st.columns([1,15])
col2.markdown("# 大连理工大学奖学金评定模型 📟")
st.sidebar.markdown("# 大连理工大学奖学金评定模型")
#st.sidebar.markdown("# 大连理工大学奖学金评定模型 📟")

#col2.title('大连理工大学奖学金评定模型')

#st.markdown("<h1 style='text-align: center; color: grey;'>111</h1>", unsafe_allow_html=True)


image=Image.open(r'./pictures\1.png')
st.image(image)

df=pd.read_csv(r"./data1.csv")
st.write('---')
st.write('✏️  第一步：请选择数据集')
#选择数据
choice = st.selectbox('   请下拉选择框', ["2021届硕士生","2022届硕士生","2023届硕士生"])



#导入表格
st.dataframe(df.iloc[0:9])
df_copy=df.copy()
#数据预处理
col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
col33.write('数据预处理结果')
df_copy.drop(axis=1,columns=["学号"],inplace=True)

df_copy=pd.get_dummies(df_copy,columns=["性别","专业","班级"])
st.dataframe(df_copy.iloc[0:9])

y=df_copy["是否获得奖学金"].values
x=df_copy.drop(axis=1,columns="是否获得奖学金").values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#选择模型 
st.write('✏️  第二步：请选择训练模型')
choice0 = st.selectbox('   请下拉选择框', ["Random Forest","SVC","K Neighbors Classifier","自定义系数"])
st.write('✏️  第三步：请设置参数')

#输出训练后的混淆矩阵和评价指标
def FinalResult(y_test, y_pred):
   image=Image.open(r'./pictures/2.png')
   st.image(image)
   st.write('---')


   #混淆矩阵
   st.write('📃  训练结果-混淆矩阵')
   col1,col2,col3=st.columns([1,1,1])

   col2.table(confusion_matrix(y_test, y_pred))
   #显示评价指标
   st.write('📃  训练结果-评价指标')
   f1 = f1_score(y_test, y_pred)
   acc=accuracy_score(y_test, y_pred)
   prec =precision_score(y_test, y_pred)
   recall =recall_score(y_test, y_pred)




   #排版
   col1,col2,col3,col4,col5,col6,col7,col8,=st.columns([1,1,1,1,1,1,1,1])
   col1.write('F1 sore:')
   col2.write(round(f1,4))
   col3.write('Accuracy:')
   col4.write(round(acc,4))
   col5.write('Precision:')
   col6.write(round(prec,4))
   col7.write('Recall:')
   col8.write(round(recall,4))
   col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
   col33.button("保存模型")





#if 'model' not in st.session_state:
if choice0 == "Random Forest":
   
   #设置参数进行模型训练
   with st.form("参数条件"):
         estimators_val = st.slider("estimators",2,200,10)
         max_depth_val = st.slider("max_depth",1,10,2)
         col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
         submitted = col33.form_submit_button("开始训练")

   st.session_state.model=RandomForestClassifier(n_estimators=estimators_val,max_depth=max_depth_val, random_state=1234)
   #训练模型
   st.session_state.model.fit(X_train, y_train)

   y_pred = st.session_state.model.predict(X_test)
   FinalResult(y_test, y_pred)
elif choice0 == 'SVC':
   with st.form("参数条件"):
         C_val = st.slider("惩罚系数C",0.,8.,0.1)
         col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
         submitted = col33.form_submit_button("开始训练")
   st.session_state.model=SVC(C=C_val)
   st.session_state.model.fit(X_train, y_train)

   y_pred = st.session_state.model.predict(X_test)
   FinalResult(y_test, y_pred)
elif choice0 == 'K Neighbors Classifier':
   with st.form("参数条件"):
         K_val = st.slider("K",1,20,1)
         col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
         submitted = col33.form_submit_button("开始训练")
   st.session_state.model = KNeighborsClassifier(n_neighbors=K_val)
   st.session_state.model.fit(X_train, y_train)

   y_pred = st.session_state.model.predict(X_test)
   FinalResult(y_test, y_pred)
elif choice0 == '自定义系数':
   with st.form("参数条件"):
   
      st.write('请设置各项评价指标系数：')
      col_1, col_2, col_3= st.columns([1, 1, 1])
      with col_1:
         learn_weight = st.number_input('学习成绩',value=0.5,min_value=0.3, max_value=0.6,step=0.02)
         keyan_weight = st.number_input('科研成果',value=0.38,min_value=0.3, max_value=0.6,step=0.02)

      with col_2:
         kechuang_weight = st.number_input('科创竞赛',value=0.02,min_value=0.01, max_value=0.2,step=0.01)
         minzhu_weight = st.number_input('民主评议',value=0.05,min_value=0.02, max_value=0.2,step=0.01)                       
                                    
      with col_3:
         shehui_weight = st.number_input('社会活动',value=0.05,min_value=0.02, max_value=0.2,step=0.01)
      col31,col32,col33,col34,col35=st.columns([1,1,1,1,1])
      submitted = col33.form_submit_button("保存模型")
   d=df_copy
   #st.write(d)
   #st.write(X_test)
   elements=[d["学习成绩"],d["科研成果"],d["科创竞赛"],d["民主评议"],d["社会活动"]]

   weights=[learn_weight,keyan_weight,kechuang_weight, minzhu_weight,shehui_weight]

   y_pred= sum([x*y for x,y in zip(elements,weights)])


   st.write(y_pred )

   

