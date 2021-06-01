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
# arrival time of ORIG-DEST. 

st.write("node_df1: ", node_df1)
nodes = node_df1.iloc[1:,:]
st.write("nodes: ", nodes)

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

