#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image



def main():
    # 初始化页面配置
    st.sidebar.markdown("# 奖学金评定系统", unsafe_allow_html=True)

    # 侧边栏
    st.sidebar.markdown("作者：苏琪", unsafe_allow_html=True)
    st.sidebar.markdown("日期：2023.7.20", unsafe_allow_html=True)



if __name__ == '__main__':
    main()
    #图标
    image=Image.open(r'./pictures/7.png')
    st.image(image)
