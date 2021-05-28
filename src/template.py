#!/usr/bin/env python
# coding: utf-8

# Use magic commands (%)

import streamlit as st

# When modifying libraries, no need to restart the kernel
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import altair as alt

alt.data_transformers.disable_max_rows()

import src.util_functions as u


# In[5]:


u.setPandasOptions()

