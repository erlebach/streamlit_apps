#!/usr/bin/env python
# coding: utf-8

# # Network Visualization with Altair
# * Without using nx_altair library
# * Test concepts on a small example



import streamlit as st
import sys
sys.path[2] = "."

## BUGs to fix

# When choosing the city LIM, the graph does not show properly. DO NOT KNOW WHY. 
# Somehow fixed. 

from src.template import *

import altair as alt

max_width = 1300
padding_left = 3
padding_right = 3
padding_top = 10
padding_bottom = 10
BACKGROUND_COLOR = 'black'
COLOR = 'lightgreen'

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

import networkx as nx
#import draw_altair as da

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
delay = st.sidebar.slider("Keep Connection times less than (min)", 0, 120, 45)

a = [3, 20, 30, 50, 70]
b = [10, 20, 30, 70, 20]

df = pd.DataFrame({'x':a, 'y':b})

#color = st.sidebar.selectbox("Node color: ", ['red','green','orange', 'blue'])
#color = col1.selectbox("Node color: ", ['red','green','orange', 'blue'])
size = st.sidebar.slider("Node size: ", 30, 400, 100)
#size = col1.slider("Node size: ", 30, 400, 100)
default_day = '2019/10/01'
day = st.sidebar.text_input("Date", value=default_day, max_chars=10)

#------------------------------
if day != default_day:
    day = default_day
    st.sidebar.write("ERROR: Date entry not yet implemented")

#st.altair_chart(chart, use_container_width=True)

# Works (using feed)
# Return dictionary
#which_handle = st.sidebar.radio("which handleCity?", ['handleCity','handleCityGraph'], index=1)

keep_early_arr = col1.checkbox("Keep early arrivals", value=False) 

which_tooltip = col1.radio("Tooltips:", ['Node','Edge','Off'], index=2)

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
    #st.write("after call handleCityGraph")

# I will need to call handleCityGraph with a specific 'id'
flight_id = "2019/10/01SJOPTY18:44796"
#st.write("before call id")
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

#st.write("after call id")
#st.write(node_df)

# Using nodes 1 (0-indexed) and beyond, determine the next flight leaving the city. 
# Given an OD: ORIG-DEST, outbound from PTY, figure out the next inbound flight to PTY. 
# Search the fsu table for DEST_ORIG, and find all flights whose departure time follows the 
# arrival time of ORIG-DEST on the same day, although days can be tricky since a flight arriving at BSB at 
# Flight departs for BSB at 20:42 (18:42 local time), and departs for PTY at 

# Given 2019/10/01PTYBSB20:42205, find the next flights. 
st.write(id_list.columns)
idd = id_list[id_list['id_nf'] == '2019/10/01PTYBSB20:42205']
st.write("idd= ", idd)

fsuu = fsu[fsu['OD'] == 'BSBPTY'][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']]
st.write("fsuu= ", fsuu)
fsuu = fsu[fsu['OD'] == 'PTYBSB'][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']]
st.write("fsuu= ", fsuu)

nodes2 = node_df1.iloc[1:]
st.write("nodes2: ", nodes2)

outbound_ids = nodes2['id'].tolist()
st.write("nodes2: ", outbound_ids)

for id in outbound_ids:
    st.write("==> id: ", id)
    #idd = id_list[id_list['id_nf'] == id]
    #st.write("idd: ", idd)
    od = id[10:16]
    st.write("od: ", od)
    fsuu = fsu[fsu['OD'] == od][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ','TAIL']]
    st.write("fsuu= ", fsuu)
    fsuu = fsu[fsu['OD'] == od][['id','OD','SCH_DEP_TMZ','SCH_ARR_TMZ','TAIL']]
    st.write("fsuu= ", fsuu)

# It will be more efficient to preprocess the fsu table (for one day or the entire table). 
# For each fight reaching a city other than PTY, compute the connecting flight (the same tail). 
# Note that some days, there are multiple tails. 
# Perhaps start with a groupby by tail? 

# Add a additional row
# Rows with flight PTY-X
def addRow(df, new_od):
    last_row = fsu.tail(1).copy()
    #st.write(last_row.shape)
    last_row['OD'] = new_od
    df = pd.concat([df, last_row])
    #st.write(df['OD'].tail())
    return df

def groupFSUbyTail(fsu0):
    fsu = fsu0.copy()
    fsu['nb_tails'] = fsu.groupby('TAIL')['id'].transform('size')
    fsu['earliest_dep'] = fsu.groupby('TAIL')['SCH_DEP_TMZ'].transform('min')
    fsu = fsu.sort_values(['TAIL', 'SCH_DEP_DTMZ'], axis=0)
    fsu = addRow(fsu, new_od='XXXXXX')
    return fsu

def groupFSUbyTail1(fsu0):
    # Here is a better approach to identifying flight connections in outside cities. 
    fsu = fsu0.copy()
    # Group by tails. Each row now includes the number of each tail 
    fsu['nb_tails'] = fsu.groupby('TAIL')['id'].transform('size')
    fsu = fsu.sort_values(['TAIL','SCH_DEP_TMZ'])
    #st.write("count values: ", fsu['nb_tails'].value_counts())

    # on Oct 1, 2019, there are: 
    # 6 tails flew once that day
    # 38 tails flew 2 times
    # 99 tails flew 3 times
    # 104 tails flew 4 times
    # 35 tails flew 5 times
    # We concentrate on the odd case

    def analyzeTailCount(fsu, cols):
        fsu_1 = fsu[fsu['nb_tails'] == 1]
        fsu_3 = fsu[fsu['nb_tails'] == 3]
        fsu_5 = fsu[fsu['nb_tails'] == 5]
        fsu_2 = fsu[fsu['nb_tails'] == 2]
        fsu_4 = fsu[fsu['nb_tails'] == 4]
        st.write(fsu_1['OD'])
        st.write("fsu_1: ", fsu_1[cols])
        st.write("fsu_3: ", fsu_3[cols])
        st.write("fsu_5: ", fsu_5[cols])
        st.write("fsu_2: ", fsu_2[cols])
        st.write("fsu_4: ", fsu_4[cols])

    cols = ['id','OD','TAIL','SCH_DEP_TMZ','SCH_ARR_TMZ']
    #analyzeTailCount(fsu, cols)

    #-------------------------------------------
    def computeFlightPairs(fsu):
        fsu1 = fsu.shift(periods=-1)
        #st.write(fsu.shape, fsu1.shape)
        #st.write("fsu: ", fsu[cols].head())
        #st.write("fsu1: ", fsu1[cols].head())
    
        flight_pairs = pd.DataFrame({'id1':fsu['id'], 'id2':fsu1['id'],
            'od1':fsu['OD'], 'od2':fsu1['OD'],
            'tail1':fsu['TAIL'], 'tail2':fsu1['TAIL'],
            'dep1':fsu['SCH_DEP_TMZ'], 'arr1':fsu['SCH_ARR_TMZ'],
            'dep2':fsu1['SCH_DEP_TMZ'], 'arr2':fsu1['SCH_ARR_TMZ']})
        flight_pairs = flight_pairs[flight_pairs['tail1'] == flight_pairs['tail2']]
        bad_flight_pairs = flight_pairs[flight_pairs['od1'].str[3:6] != flight_pairs['od2'].str[0:3]]
        st.write("flight_pairs: ", flight_pairs)
        st.write("bad_pairs: ", bad_flight_pairs)
    
        # Apparently, 
        # Tail HP1722 was transferred from SJO to TGU on 2019/10/01
        # Tail HP1829: not clear. But something happened on 2019/10/01
        # Tail HP1857 was transferred from TGU to SJO on 2019/10/01
        tail1722 = fsu[fsu['TAIL'] == 'HP1722']
        tail1829 = fsu[fsu['TAIL'] == 'HP1829']
        tail1857 = fsu[fsu['TAIL'] == 'HP1857']

        st.write("tail1722: ", tail1722[cols])
        st.write("tail1829: ", tail1829[cols])
        st.write("tail1857: ", tail1857[cols])

        # A flight arrives at SJO at 14:16 pm (Zulu), HP1722. 
        # Two different nearest flights depart SJO at 14:25 and 17:28. 
        # They are different tails. So their departures can only be affected
        # by incoming pax. 
        fsu_sjo = fsu[fsu['OD'] == 'SJOPTY']
        st.write("fsu_sjo: ", fsu_sjo[cols].sort_values('SCH_DEP_TMZ'))
        return flight_pairs, bad_flight_pairs

    flight_pairs, bad_pairs = computeFlightPairs(fsu)


    #-------------------------
    def getInbounds(df, flight_id):
        """
        Given a flight_id (string) to city X, return all flights from city X
        back to PTY, sorted chronologically
        """
        inbound_od = flight_id[13:16] + 'PTY'
        fsu_X = df[df['OD'] == inbound_od]

        if fsu_X.shape[0] == 0:
            st.write("<<<< fsu_X.shape[0]: ", fsu_X.shape[0])
            return None

        fsu_X = fsu_X.sort_values('SCH_DEP_TMZ')
        return fsu_X

    #-------------------------
    def findNextDepartures(f_od, pty_outbound_id, pty_inbounds):
        """
        Return the two departures closest to the arrival time of pty_outbound_id
        Return two full records (or only one if only one is available
        """
        arr_time = pty_outbound_id[16:20]
        lst = pty_inbounds[pty_inbounds['SCH_DEP_TMZ'] > arr_time]
        if lst.shape[0] <= 2:
            return lst
        else:
            return lst.iloc[0:2]

    # pty_outbound_id NOT DEFINED
    lst = findNextDepartures(df, pty_outbound_id, pty_inbounds)
    st.write("lst: ", lst[cols])
    return lst

    #-------------------------
    def inboundOutboundDict(f_od, fsu_ids):
        dct = {}
        for i in range(fsu_ids.shape[0]):
            outbound_id = fsu_ids[i]
            pty_inbounds = getInbounds(fsu, outbound_id)
            # 1 or 2 next departures
            next_departures = findNextDepartures(f_od, outbound_id, pty_inbounds)
            dct[outbound_id] = next_departures
        return dct

    # Precalculate (for one day) the 1-2 flights from city X following the inbound flight. The assumption is that one of these will be the correct flight to follow. Ideally, the tail should be the same as the incoming flight. 

    # Given an OD pair PTY-X, list all departures in ascending order, Zulu.
    f_od = fsu[cols].sort_values(['OD','SCH_DEP_TMZ'])
    st.write("f_od: ", f_od)

    fsu_ids = f_od[f_od['OD'].str[0:3] == 'PTY']['id'].values
    st.write("=========================")
    st.write("fsu_ids.shape: ", fsu_ids.shape)
    dct = inboundOutboundDict(f_od, fsu_ids)


    st.stop()

    # For each group, add a row if there is an odd number of tails
    # 1 flight for a tail per day
    # 3 flights for a tail per day
    # 5 flights for a tail per day
    # 7 flights for a tail per day
    fsu['earliest_dep'] = fsu.groupby('TAIL')['SCH_DEP_TMZ'].transform('min')
    fsu = fsu.sort_values(['TAIL', 'SCH_DEP_DTMZ'], axis=0)
    fsu = addRow(fsu, new_od='XXXXXX')

# Following a tail throughout the day is now very easy. 

# What I really need (for an entire day). 
# For each outbound flight to city X, determine the next flight leaving X for PTY with the 
# same tail. On occasions, tails will be changed, but this is ignored. 
# 
# The algorithm is simple and as follows: 
# 1) identify all rows with a flight PTY-X
# 2) The desired flight is the next row. On occasion, the next row will not match. 
#    We will deal with that later. 
#    Worry about edge effects. For example, the last row will not have a next row. Easiest
#    solution: Copy the last row to create a new row, and make the OD='XXXPTY'

# TRacking by tail works most of the time, but not all the time. 
# I need a better algorithm to find connections at outside cities (non-PTY)

next_departures = groupFSUbyTail1(fsu)
st.write("next_departures")
st.write(next_departures)
st.stop()

fsu = groupFSUbyTail(fsu)
fsu = fsu.reset_index(drop=True)
st.write("fsu: ", fsu['TAIL'])
indices  = fsu.index[fsu['OD'].str[0:3] == 'PTY']
st.write(indices)

fsu1 = fsu.iloc[indices]
fsu2 = fsu.iloc[indices+1,:]   #<<< ERROR

st.write("fsu.shape: ", fsu.shape)
st.write("fsu1.shape: ", fsu1.shape)
st.write("fsu2.shape: ", fsu2.shape)

st.write("fsu: ",  fsu[['TAIL','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']])
st.write("fsu1: ",  fsu1[['TAIL','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']])

# Concatenate f2 and f3 horizonally (axis=1)
# Given a flight id, I want the connecting flight ID
#fsu4 = pd.concat([fsu1, fsu2], axis=1) # not done correctly
#st.write("fsu4= ", fsu4['id'])
#st.write("fsu4.shape= ", fsu4.shape)


# Dictionary only computed once
def computeDict(fsu1, fsu2):
    dct = {}
    nodct = {}

    ids_PTYX = fsu1[['id','TAIL']]
    ids_XPTY = fsu2[['id','TAIL']]
    st.write("ids_PTYX= ", ids_PTYX)
    st.write("ids_XPTY= ", ids_XPTY)

    ptyx = ids_PTYX['id'].tolist()
    xpty = ids_XPTY['id'].tolist()
    
    for i in range(ids_PTYX.shape[0]):
        if ptyx[i][13:16] == xpty[i][10:13]:
            dct[ptyx[i]] = xpty[i]
        else: 
            nodct[ptyx[i]] = xpty[i]
    return dct, nodct

dct,nodct = computeDict(fsu1, fsu2)
st.write("dct items")
st.write(dct)
st.write("nodct items")
st.write(nodct)


st.stop()

# 70 elements out of 120
st.write("dict length: ", len(dct))
st.write("nodict length: ", len(nodct))

st.stop()

fsu['nb_tails'] = fsu.groupby('TAIL')['id'].transform('size')
fsu['earliest_dep'] = fsu.groupby('TAIL')['SCH_DEP_TMZ'].transform('min')
st.write(fsu[['id','nb_tails']])
# It appears that the first flight of each tail is an inbound flight into PTY. I wonder if that is always true.
# Interest rows are the ones with 3,5,7 tails. I would expect the number of flights with a given tail to be an even number.

st.write(fsu.columns)
st.write("\n3 flights per day with the same tail")
fss = fsu[fsu['nb_tails'] == 3].reset_index()[['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD','SCH_DEP_DTMZ']]
st.write(fss.columns)

fss = fss.sort_values(['TAIL', 'SCH_DEP_DTMZ'], axis=0)
st.write("fss: ", fss)

# I find that when there are three flights per day with a given tail, 
# in all cases (except for 3), the flight originates outside PTY. In 
# three cases, the flight originates in PTY. At least on Oct. 01, 2019.
st.write(fss.iloc[range(0,fss.shape[0],3),:][['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD']])

st.write("\n5 flights per day with the same tail")
fss = fsu[fsu['nb_tails'] == 5].reset_index()[['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD','SCH_DEP_DTMZ']]
st.write(fss.columns)
fss = fss.sort_values(['TAIL', 'SCH_DEP_DTMZ'], axis=0)
st.write("fss: ", fss)

# I find that when there are five flights per day with a given tail, 
# in all cases, the flight originates outside PTY. 
# There are 5 flights for only HAV, MDE, VVI, GYE, SDQ, POS, POA
st.write(fss.iloc[range(0,fss.shape[0],5),:][['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD']])

st.write("7 flights per day with the same tail: None")
fss = fsu[fsu['nb_tails'] == 7].reset_index()[['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD','SCH_DEP_DTMZ']]

st.write(fss.iloc[range(0,fss.shape[0],7),:][['id','FLT_NUM','TAIL','nb_tails','earliest_dep','SCH_DEP_TMZ','SCH_ARR_TMZ','OD']])



#------------------------------------------------------------

# Create Altair Chart

# Allow tooltip to work in full screen mode (expanded view)
# From: https://discuss.streamlit.io/t/tool-tips-in-fullscreen-mode-for-charts/6800/9
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
             unsafe_allow_html=True)

def createStepLines(node_df, edge_df, edge_labels=('id_f_y','id_nf_y')):
        """
        Arguments
            node_df: a dataframe with at least the three column labels 'id', 'x', 'y'. The id can be a string.
            edge_df: a dataframe with two column names; 'id_f_y' (source), 'id_nf_y' (target). 
            edge_labels: a tuple with the names of the edge source and destination labels. Default: ('id_f_y','id_nf_y')
        Return 
            a database with columns x0,y0,x1,y1, etc where (xi,yi) are the coordinates of an edge.
        """
        # Set up step (horizontal/vertical segments) lines
        # For each edge, create a set of nodes, stored in a special dataframe
        # For now, use loops since there are not many edges

        e1 = edge_df[edge_labels[0]]
        e2 = edge_df[edge_labels[1]]

        # x,y of node 1 and 2 of each edge.
        # Construct intermediate points (ids1a and ids1b)
        ids1 = node_df.set_index('id').loc[e1.tolist()].reset_index()[['x','y']]
        ids2 = node_df.set_index('id').loc[e2.tolist()].reset_index()[['x','y']]
        ids1a = pd.DataFrame([ids1.x, 0.5*(ids1.y+ids2.y)]).transpose()
        ids1b = pd.DataFrame([ids2.x, 0.5*(ids1.y+ids2.y)]).transpose()
 
        df_step = pd.concat([ids1, ids1a, ids1b, ids2], axis=1)
        df_step.columns = ['x1','y1','x2','y2','x3','y3','x4','y4']

        # Now create one line per edge: 
        #  col 1: [x1,x2,x3,x4].row1
        #  col 2: [y1,y2,y3,y4].row1
        #  col 3: [x1,x2,x3,x4].row2
        #  col 4: [y1,y2,y3,y4].row2
        df_step_x = df_step[['x1','x2','x3','x4']].transpose()
        df_step_y = df_step[['y1','y2','y3','y4']].transpose()

        # relabel the columns of df_step_x as 'x0', 'x1', ..., 'x15'
        # relabel the columns of df_step_y as 'y0', 'y1', ..., 'y15'
        df_step_x.columns = ['x'+str(i) for i in range(df_step_x.shape[1])]
        df_step_y.columns = ['y'+str(i) for i in range(df_step_y.shape[1])]

        df_step_x = df_step_x.reset_index(drop=True)
        df_step_y = df_step_y.reset_index(drop=True)

        df_step = pd.concat([df_step_x, df_step_y], axis=1)

        # Create a dataframe
        df_step = df_step.reset_index()  # generates an index column
        return df_step


#----------------------------------------------------------------------------
def drawStepEdges(df_step):
        """
        Arguments
            df_step: An array where each column is either x or y coordinate of an edge composed of four points. 
            The column names must be x0,y0,x1,y1, ... not in any particular order. 
            An additional column labeled 'index' specifies the order of the nodes on each edge.

        Return
            The layer chart.
        """
        # Technique found at https://github.com/altair-viz/altair/issues/1036
        base = alt.Chart(df_step)
        layers = alt.layer(
            *[  base.mark_line(
                color='red',
                opacity=1.0,
                strokeWidth=2
            ).encode(
                x=alt.X('x'+str(i), axis=alt.Axis(title="Outbounds")), 
                y=alt.Y('y'+str(i), axis=alt.Axis(title="", labels=False)),
                order='index'
            ) for i in range(int(df_step.shape[1]/2))
            ], data=df_step)
        return layers
#--------------------------------------------------

def drawPlot3(node_df, edge_df, which_tooltip):
    nb_nodes = node_df.shape[0]
    nb_edges = edge_df.shape[0]

    # By convention, the first node is the feeder. 
    x = []
    nb_outbounds = nb_nodes - 1
    x.append(0.5) # centered
    x.extend(np.linspace(0, 1, nb_outbounds))
    node_df['x'] = x
    node_df['y'] = 0.8
    node_df.loc[1:,'y'] = 0.3  # assumes node_df is ordered correctly (feeder first)

    # Create and draw edges as a series of horizontal and vertical lines
    df_step = createStepLines(node_df, edge_df)
    layers = drawStepEdges(df_step)

    # Set up tooltips searching via mouse movement
    node_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['x','y'], empty='none')

    edge_nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['x_mid','y_mid'], empty='none')

    lookup_data = alt.LookupData(
        node_df, key="id", fields=["x", "y"]
    )

    lookup_data = alt.LookupData(
        node_df, key="id", fields=["x", "y"]
    )

    nodes = alt.Chart(node_df).mark_rect(
        width=50,
        height=20,
        opacity=1.0,
        align = 'center',
    ).encode(
        x = alt.X('x:Q'),
        y = 'y:Q',
        color = 'arr_delay',
    )

    node_tooltips = alt.Chart(node_df).mark_circle(
        size=500,
        opacity=0.0,
    ).encode(
        x = 'x:Q',
        y = 'y:Q',
        tooltip=['id','arr_delay','dep_delay','od']
    )

    node_text = alt.Chart(node_df).mark_text(
        opacity = 1.,
        color = 'black',
        align='center',
        baseline='middle'
    ).encode(
        x = 'x:Q',
        y = 'y:Q',
        text='od',
        size=alt.value(10)
    )

    # Create a data frame with as many columns as there are edges. 
    # Each column is four points. 
    edges = alt.Chart(edge_df).mark_rule(
        strokeOpacity=.1,
        stroke='yellow',
    ).encode(
        x = 'x:Q',
        y = 'y:Q',
        x2 = 'x2:Q',
        y2 = 'y2:Q',
        stroke = 'pax:Q',
        strokeWidth = 'pax:Q' #'scaled_pax:Q'
    ).transform_lookup(
        # extract all flights with 'origin' from airports (state, lat, long)
        lookup='id_f_y',   # needed to draw the line. 'origin' is in flights_airport.csv
        from_=lookup_data
    ).transform_lookup(
        # extract all flights with 'destination' from airports (state, lat, long) renamed (state, lat2, long2)
        lookup='id_nf_y',
        from_=lookup_data,
        as_=['x2', 'y2']
    )

    mid_edges = alt.Chart(edge_df).mark_circle(color='yellow', size=100, opacity=0.1).encode(
        x = 'mid_x:Q',
        y = 'mid_y:Q',
        tooltip= ['avail','planned','delta','pax']
    ).transform_lookup(
        # extract all flights with 'origin' from airports (state, lat, long)
        lookup='id_f_y',   # needed to draw the line. 'origin' is in flights_airport.csv
        from_=lookup_data
    ).transform_lookup(
        # extract all flights with 'destination' from airports (state, lat, long) renamed (state, lat2, long2)
        lookup='id_nf_y',
        from_=lookup_data,
        as_=['x2', 'y2']
    ).transform_calculate(
        mid_x = '0.5*(datum.x2 + datum.x)',
        mid_y = '0.5*(datum.y2 + datum.y)'
    )

    if which_tooltip == 'Edge':
        #col1.write("add edge tip")
        mid_edges = mid_edges.add_selection(
            edge_nearest
        )
    elif which_tooltip == 'Node':
        #col1.write("add node tip")
        node_tooltips = node_tooltips.add_selection(
            node_nearest
        )
    elif which_tooltip == 'Off':
        #col1.write("no tooltips")
        pass

    full_chart = (layers + edges + nodes + node_text + node_tooltips + mid_edges)

    # Chart Configuration
    full_chart = full_chart.configure_axisX(
        labels=False,
    )

    return full_chart


# Experimental. Use drawPlot2 for stable results
if edge_df.shape[0] == 0:
    col2.write("Nothing to display: all passengers have sufficient connection time")
else:
    # drawPlot2: edges are still oblique
    # drawPlot3: edges are step functions, horizontal/vertical
    chart2 = drawPlot3(node_df, edge_df, which_tooltip)
    col2.altair_chart(chart2, use_container_width=True)

st.stop()



#----------------------------------------------
@st.cache
def createGraph(nb_nodes, nb_edges):
    e1 = randint(nb_nodes, size=nb_edges)
    e2 = randint(nb_nodes, size=nb_edges)
    G = nx.DiGraph()

    for i in range(nb_nodes):
        G.add_node(i)

    for i,j in zip(e1,e2):
        G.add_edge(i,j)
    return G

G = createGraph(nb_nodes, nb_edges)

#----------------------------------------------
def drawPlot1(G):
    # Start from Graph to create Data Frames
    #   (normally, one would start with the data frame)
    e1 = map(lambda x: x[0], G.edges())
    e2 = map(lambda x: x[1], G.edges())
    ed1 = [] 
    ed2 = []
    for i,j in zip(e1,e2):
        ed1.append(i)
        ed2.append(j)
    
    edges_df = pd.DataFrame({'e1':ed1, 'e2':ed2})
    edges_df = edges_df.reset_index().rename(columns={'index':'id'})
    
    nodes_df = pd.DataFrame({'id':list(range(0,nb_nodes))})
    
    pos = normal(nb_nodes,2)
    
    nodes_df['x'] = pos[:,0]
    nodes_df['y'] = pos[:,1]
    # nodes: data frame with id, x, y
    
    # Create an Altair graph without ever creating a NetworkX graph. An actual graph structure would be necessary to compute graph metrics in certain circumstances. 
    edges_df['e1'].dtype
    
    edges_df['e1'] = edges_df['e1'].astype('int')
    edges_df['e2'] = edges_df['e2'].astype('int')
    
    edges_df.shape, nodes_df.shape
    
    lookup_data = alt.LookupData(
        nodes_df, key="id", fields=["x", "y"]
    )
    
    node_brush = alt.selection_interval()
    
    nodes = alt.Chart(nodes_df).mark_circle(size=10).encode(
        x = 'x:Q',
        y = 'y:Q',
        color = 'y:N',
        tooltip = ['x','y']
    ).add_selection(
        node_brush
    )
    
    edges = alt.Chart(edges_df).mark_rule().encode(
        x = 'x:Q',
        y = 'y:Q',
        x2 = 'x2:Q',
        y2 = 'y2:Q',
        color = 'x2:Q'
    ).transform_lookup(
        # extract all flights with 'origin' from airports (state, lat, long)
        lookup='e1',   # needed to draw the line. 'origin' is in flights_airport.csv
        from_=lookup_data
    ).transform_lookup(
        # extract all flights with 'destination' from airports (state, lat, long) renamed (state, lat2, long2)
        lookup='e2',
        from_=lookup_data,
        as_=['x2', 'y2']
    ).transform_filter(
        node_brush
    ).properties(width=1000, height=800)
        
    full_chart = (edges + nodes)
    return full_chart

chart1 = drawPlot1(G)
#col2.chart(chart1)
col2.altair_chart(chart1, use_container_width=True)
#----------------------------------------------------------------------

# Chart using Copa data

