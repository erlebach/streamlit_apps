#!/usr/bin/env python
# coding: utf-8

# # Network Visualization with Altair
# * Without using nx_altair library
# * Test concepts on a small example



import src.altair_support as altsup
import networkx as nx
import sys
import traceback
from src.template import *
sys.path[2] = "."


# When choosing the city LIM, the graph does not show properly. DO NOT KNOW WHY. 
# Somehow fixed. 

# Set parameters inside this function
altsup.allowTooltipsInExpandedWindows()

normal = np.random.rand
randint = np.random.randint

# * Create routines to create the files below with the '_date' section.
# * Look at the noteboo prepare_ids_for_visualization.ipynb
# * I should be able to work on a range of dates. 

st.sidebar.empty()
col1, col2 = st.beta_columns([4,12])

bookings_f = pd.read_csv("my_data/bookings_date_f.csv.gz")
bookings_nf = pd.read_csv("my_data//bookings_date_nf.csv.gz")
feeders = pd.read_csv("my_data/bookings_idlist_pax_date.csv.gz")
fsu = pd.read_csv("my_data/node_attributes_df.csv.gz")
id_list = pd.read_csv("my_data/id_list_date.csv.gz")

#col1.write("---")
# control color via CSS
#col1.write(f"{fsu.shape}, {id_list.shape}, {feeders.shape}, {bookings_nf.shape}, {bookings_f.shape}")
#st.write(f"{fsu.shape}, {id_list.shape}, {feeders.shape}, {bookings_nf.shape}, {bookings_f.shape}")
#st.sidebar.write(f"{fsu.shape}, {id_list.shape}, {feeders.shape}, {bookings_nf.shape}, {bookings_f.shape}")
#st.write(fsu.shape, id_list.shape, feeders.shape, bookings_nf.shape, bookings_f.shape)
#st.sidebar.write(fsu.shape, id_list.shape, feeders.shape, bookings_nf.shape, bookings_f.shape)

#id_list['nb_outbounds'] = id_list.groupby('id_f')['id_nf'].transform('size')
#id_list.drop("nb_outbounds", axis=1, inplace=True)

def readFullFeed():
    feed = pd.read_csv("my_data/bookings_ids+pax.csv")
    col1(f"feed.shape: {feed.shape}")
    return feed

def createDayFeed(feed, day):
    feed1 = feed[(feed['id_f'].str.contains(day)) | (feed['id_nf'].str.contains(day))]
    feed1.to_csv("my_data/bookings_ids+pax_date.csv.gz", index=0)

def readDayFeed():
    feed = pd.read_csv("my_data/bookings_ids+pax_date.csv.gz")
    return feed

#feed = readFullFeed()
#createDayFeed(feed, day)

day = '2019/10/01'
feed = readDayFeed()
#col1.write(f"feed.shape: {feed.shape}")


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

cities = fsu[fsu['OD'].str[0:3] != 'PTY'].loc[:,'OD'].str[0:3].unique()

# find the index for SJO
# convert to int from np.int64 (not the same)
init_ix = int(np.where(cities == 'SJO')[0][0])

which_city = st.sidebar.selectbox("Select a City: ", cities, index=init_ix)
delay = st.sidebar.slider("Keep Connection times less than (min)", 0, 120, value=70)

xmax = st.sidebar.slider("Domain size in X", 0, 5, 2)

pty_feeders = fsu[fsu['id'].str[10:13] == which_city]['id'].sort_values().to_list()
which_fid = st.sidebar.selectbox("Select a feeder: ", pty_feeders) #, index=init_fid)

rect_color = st.sidebar.selectbox("Node color: ", ['yellow','green','red','blue','darkgray','black'], index=3)

text_color = st.sidebar.selectbox("Text color: ", ['yellow','green','red','blue','darkgray','black'], index=0)

size = st.sidebar.slider("Node size: ", 30, 400, 100)
default_day = '2019/10/01'
day = st.sidebar.text_input("Date", value=default_day, max_chars=10)
if day == default_day:
    print("correct day!!!")

#------------------------------
if day != default_day:
    day = default_day
    st.sidebar.write("ERROR: Date entry not yet implemented")

#st.altair_chart(chart, use_container_width=True)

# Works (using feed)
# Return dictionary
#which_handle = st.sidebar.radio("which handleCity?", ['handleCity','handleCityGraph'], index=1)

which_tooltip = col1.radio("Tooltips:", ['Node','Edge','Off'], index=2)

keep_early_arr = col1.checkbox("Keep early arrivals", value=True) 

"""
which_handle = 'handleCityGraph'
if which_handle == 'handleCity':
    dfs = u.handleCity(
            which_city, 
            'all', 
            id_list, 
            fsu, 
            bookings_f, 
            feed, 
            is_print=True, 
            delay=delay
    )
else:
    node_df, edge_df = u.handleCityGraph(
            keep_early_arr, 
            which_city, 
            'all', 
            id_list, 
            fsu, 
            bookings_f, 
            feed, 
            is_print=False, 
            delay=delay
    )
"""


# I will need to call handleCityGraph with a specific 'id' (node id)
flight_id = "2019/10/01SJOPTY18:44796"
flight_id = which_fid
#flight_id = '2019/10/01MDEPTY10:20532'  # FIGURE OUT coord issue!!
node_df1, edge_df1 = u.handleCityGraphId(
            flight_id,
            keep_early_arr, 
            id_list, 
            fsu, 
            bookings_f, 
            feed, 
            is_print=False, 
            delay=delay
        )

# Using nodes 1 (0-indexed) and beyond, determine the next flight leaving the city. 
# Given an OD: ORIG-DEST, outbound from PTY, figure out the next inbound flight to PTY. 
# Search the fsu table for DEST_ORIG, and find all flights whose departure time follows the 
# arrival time of ORIG-DEST on the same day, although days can be tricky since a flight arriving at BSB at 
# Flight departs for BSB at 20:42 (18:42 local time), and departs for PTY at 

# Given 2019/10/01PTYBSB20:42205, find the next flights. 
#st.write(id_list.columns)
#st.write(node_df1)
idd = id_list[id_list['id_nf'] == '2019/10/01PTYBSB20:42205']

fsuu = fsu[fsu['OD'] == 'BSBPTY'][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']]
fsuu = fsu[fsu['OD'] == 'PTYBSB'][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']]

nodes2 = node_df1.iloc[1:]
outbound_ids = nodes2['id'].tolist()

if False:
  for id in outbound_ids:
    st.write("==> id: ", id)
    #idd = id_list[id_list['id_nf'] == id]
    st.write("idd: ", idd)
    od = id[10:16]
    st.write("od: ", od)
    fsuu = fsu[fsu['OD'] == od][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ','TAIL']]
    st.write("fsuu= ", fsuu)
    fsuu = fsu[fsu['OD'] == od][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ','TAIL']]
    st.write("fsuu= ", fsuu)

#---------------------------------------------------------

#-------------------------------------------

# Concatenate f2 and f3 horizonally (axis=1)
# Given a flight id, I want the connecting flight ID
#fsu4 = pd.concat([fsu1, fsu2], axis=1) # not done correctly
#st.write("fsu4= ", fsu4['id'])
#st.write("fsu4.shape= ", fsu4.shape)

# Given a flightId of flight F, return the closest connecting two flights with the departures later than and closest to F's scheduled arrival time. 
dct = altsup.groupFSUbyTail1(fsu)
#st.write("dct= ", dct)

#tail_list = altsup.computeFlightPairs(fsu)

def tailDict(fsu):
    """
    Return a DataFrame, indexed by incoming flight id. 
    The incoming ID can be that of an inbound or outbound flight to PTY 
    """
    fsu = fsu.sort_values(['TAIL','SCH_DEP_TMZ'])
    st.write(fsu[['TAIL','SCH_DEP_TMZ']])
    fsu1 = fsu.shift(periods=-1)
    flight_pairs = pd.DataFrame({'id1':fsu['id'], 'id2':fsu1['id'],
        'od1':fsu['OD'], 'od2':fsu1['OD'],
        'tail1':fsu['TAIL'], 'tail2':fsu1['TAIL'],
        'dep1':fsu['SCH_DEP_TMZ'], 'arr1':fsu['SCH_ARR_TMZ'],
        'dep2':fsu1['SCH_DEP_TMZ'], 'arr2':fsu1['SCH_ARR_TMZ']})
    flight_pairs = flight_pairs[flight_pairs['tail1'] == flight_pairs['tail2']]
    return flight_pairs.set_index('id1', drop=False)

pairs = tailDict(fsu)
st.write(pairs)



# Why is FSU required? 
# The flight_id_level of the first node to add to the graph
#st.write("First tier, before return flights: node_df1= ", node_df1)
# some OD's missing. How
node_df1, edge_df1 = u.getReturnFlights(pairs, node_df1, edge_df1, dct, fsu, flight_id_level=2)
#st.write("First tier, after return flights: node_df1= ", node_df1)

# Get second tier for flights
#st.write("First tier: node_df1= ", node_df1)
ids = node_df1[node_df1['lev'] == 2]['id'].to_list()
st.write("second tier: ", ids)

node_dct = {}
edge_dct = {}

# Next level of flights
if False:
  for fid in ids:
    st.write("fid: ", fid)
    #try:
    if 1:
        node_df2, edge_df2 = u.handleCityGraphId(
            fid,
            keep_early_arr, 
            id_list, 
            fsu, 
            bookings_f, 
            feed, 
            is_print=False, 
            delay=delay,
            flight_id_level=2,
        )
    #except:
        #st.write("Error in handleCityGraphId")
        #st.write(traceback.print_exc())
        #continue

    # Not sure what this is for
    node_dct[fid] = node_df2
    edge_dct[fid] = edge_df2

    st.write("node_df2: ", node_df2)
    
    # The first row is the root node. So I can simply append these
    # to the main node and edge structures

    # Inefficient perhaps
    node_df1 = pd.concat([node_df1,node_df2.iloc[1:]])
    edge_df1 = pd.concat([edge_df1,edge_df2])


#st.write("after getReturnFlights, node_df: ", node_df1)

# Create a dictionary of node/edge pairs
# I would like to connect the outbound city X to the inbounds from city X by a dotted line.

#-----------------------------------------------------
# SERIOUS BUG: The x/y plots not potting properly. Perhaps there are
# too many nodes to plot?  This is MDE

node_df = node_df1.copy()
edge_df = edge_df1.copy()

# Experimental. Use drawPlot2 for stable results
if edge_df.shape[0] == 0:
    col2.write("Nothing to display: all passengers have sufficient connection time")
else:
    # drawPlot2: edges are still oblique
    # drawPlot3: edges are step functions, horizontal/vertical
    chart3 = altsup.drawPlot3(node_df, edge_df, which_tooltip, xmax, rect_color, text_color)
    col2.altair_chart(chart3, use_container_width=True)

st.stop()



