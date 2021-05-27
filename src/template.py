#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Use magic commands (%)

import streamlit as st

# When modifying libraries, no need to restart the kernel
try:
    st.write("try")
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    st.write("except")
    pass


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import altair as alt

alt.data_transformers.disable_max_rows()

import src.copa.util_functions as u


# In[5]:


u.setPandasOptions()

