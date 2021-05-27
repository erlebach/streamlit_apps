#!/usr/bin/env python
# coding: utf-8

# # Network Visualization with Altair
# * Without using nx_altair library
# * Test concepts on a small example



import streamlit as st
import sys
st.write(sys.path)
sys.path[1] = "."

from src.template import *

import altair as alt

"""
from core import *
#from src.copa.core import *
"""

import networkx as nx
#import draw_altair as da

normal = np.random.rand
randint = np.random.randint

# * Create routines to create the files below with the '_date' section.
# * Look at the noteboo prepare_ids_for_visualization.ipynb
# * I should be able to work on a range of dates. 

st.sidebar.empty()

bookings_f = pd.read_csv("my_data/bookings_date_f.csv.gz")
bookings_nf = pd.read_csv("my_data//bookings_date_nf.csv.gz")
feeders = pd.read_csv("my_data/bookings_idlist_pax_date.csv.gz")
fsu = pd.read_csv("my_data/node_attributes_df.csv.gz")
id_list = pd.read_csv("my_data/id_list_date.csv.gz")

#id_list = pd.read_csv("../data/id_list.csv.gz")

#ff = id_list[id_list['id_f'].str.contains('2019/10/01')]
#ff.to_csv("../data/id_list_date.csv.gz")

#st.write("ff.shape: ", ff.shape)
#st.write(ff.head())

st.write("---")
st.write(fsu.shape, id_list.shape, feeders.shape, bookings_nf.shape, bookings_f.shape)

#id_list['nb_outbounds'] = id_list.groupby('id_f')['id_nf'].transform('size')
#id_list.drop("nb_outbounds", axis=1, inplace=True)

def readFullFeed():
    feed = pd.read_csv("../data/bookings_ids+pax.csv")
    st.write(f"feed.shape: {feed.shape}")
    return feed

def createDayFeed(feed, day):
    feed1 = feed[(feed['id_f'].str.contains(day)) | (feed['id_nf'].str.contains(day))]
    feed1.to_csv("../data/bookings_ids+pax_date.csv.gz", index=0)

def readDayFeed():
    feed = pd.read_csv("../data/bookings_ids+pax_date.csv.gz")
    return feed

#feed = readFullFeed()
#createDayFeed(feed, day)

day = '2019/10/01'
feed = readDayFeed()
st.write("feed.shape: ", feed.shape)


#feeders[feeders['id_nf'] == '2019/10/01PTYADZ14:46610']['pax_f'].sum()
#bookings_nf[bookings_nf['id_nf'] == '2019/10/01PTYADZ14:46610']

# ## Calculate available connection time of all feeders
# * Step 0: choose an incoming feeder flight
# * step 1: compute arrival times of all feeders
# * step 2: compute scheduled departure time of all outgoing flights
# * Step 3: compute available connection time for feeder passengers
# * Step 4: only keep feeders with connection time < X min (X chosen to be 45 min or 60 min)
# * Step 5: draw

f = feeders[['id_f','id_nf','pax_f','pax_nf']]

#feeders[feeders['id_f'] == '2019/10/01MVDPTY04:23284']
#feed.columns  # 611,000 rows
#feed[feed['id_f'] == '2019/10/01MVDPTY04:23284']
#bookings_f[bookings_f['id_f'] == '2019/10/01MVDPTY04:23284']

cities = fsu[fsu['OD'].str[0:3] != 'PTY'].loc[:,'OD'].str[0:3].unique()

which_city = st.sidebar.selectbox("Select a City: ", cities)
delay = st.sidebar.slider("Delay", 0, 120, 45)

a = [3, 20, 30, 50, 70]
b = [10, 20, 30, 70, 20]

df = pd.DataFrame({'x':a, 'y':b})

color = st.sidebar.selectbox("Node color: ", ['red','green','orange', 'blue'])
size = st.sidebar.slider("Node size: ", 30, 400, 100)
default_day = '2019/10/01'
day = st.sidebar.text_input("Date", value=default_day, max_chars=10)
#streamlit.text_input(label, value='', max_chars=None, key=None, type='default', help=None)

if day != default_day:
    day = default_day
    st.sidebar.write("ERROR: Date entry not yet implemented")

chart = alt.Chart(df).mark_circle(color=color, size=size).encode(
    x = 'x',
    y = 'y',
    tooltip = ['x','y']
)

st.altair_chart(chart, use_container_width=True)
st.write("feeders: ", feeders.shape)
st.write(f"before handleCity: {which_city}")

# Works (using feed)
output = u.handleCity(which_city, 'all', id_list, fsu, bookings_f, feed, is_print=True, delay=delay)
#output = u.handleCity(which_city, 'all', id_list, fsu, bookings_f, feed1, is_print=True, delay=delay)
# Does not work
#output = u.handleCity(which_city, 'all', id_list, fsu, bookings_f, feeders, is_print=True)

st.stop()

st.write("Loop over all cities")
for c in cities:
    #print("-------------------------------------------")
    output = u.handleCity(c, 'all', fsu, id_list, bookings_f, feeders, is_print=True)
    #st.write(output)

#output = u.handleCity(which_city, 'all', fsu, id_list, bookings_f, feeders, is_print=True)
st.write(output)

st.write(f"# Delay information regarding city {which_city}")

st.write(output)
