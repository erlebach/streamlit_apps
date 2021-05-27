import streamlit as st
import pandas as pd
import sys

st.write(sys.path)

st.write("# hello")
st.write("## Bye")
a = 5
f'''
## What is this? 
### Nothing {a}
- my list
'''

df = pd.DataFrame({'x':[1,2,3], 'y':[3,6,12]})
st.write(df)
st.dataframe(df)

# Not clear whether cache is having any effect
@st.cache
def readFSU(file):
    df = pd.read_csv(file) #"copa_data/dff_agg1_correct_pax.csv")
    return df

fn = "copa_data/FSU_fully_cleaned.csv.gz/"
df = readFSU(fn)
st.dataframe(readFSU(fn))

"""
Finished
"""
