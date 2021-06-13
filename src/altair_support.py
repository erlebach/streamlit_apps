from src.template import *
from collections import defaultdict
from collections import deque
import networkx as nx
from node import Node 
from walker import Walker
#from GE_walker import WalkerGE

def allowTooltipsInExpandedWindows():
    # Need a better way to set these. Perhaps in a dictionary set as 
    # an argument? 
    max_width = 1300
    padding_left = 3
    padding_right = 3
    padding_top = 10
    padding_bottom = 10
    BACKGROUND_COLOR = 'black'
    COLOR = 'lightgreen'

    st.markdown(f"""
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
    """, unsafe_allow_html=True,)

#----------------------------------------------------------
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

        ## REASON FOR ERROR: PTY-CUN does not have a return flight to PTY .
        # There is an error in edge array. There is one too many edges.
        # Must fix in getReturnFlights


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
def drawStepEdges(df_step, scale):
        """
        Arguments
            df_step: An array where each column is either x or y coordinate of an edge composed of four points. 
            The column names must be x0,y0,x1,y1, ... not in any particular order. 
            An additional column labeled 'index' specifies the order of the nodes on each edge.
            scale: scale parameter for xaxis

        Return
            The layer chart.
        """
        # Technique found at https://github.com/altair-viz/altair/issues/1036
        base = alt.Chart(df_step)
        layers = alt.layer(
            *[  base.mark_line(
                color='red',
                opacity=0.5,
                strokeWidth=2
            ).encode(
                x=alt.X('x'+str(i), axis=alt.Axis(title="Outbounds"), scale=scale), 
                y=alt.Y('y'+str(i), axis=alt.Axis(title="", labels=False)),
                order='index'
            ) for i in range(int(df_step.shape[1]/2))
            ], data=df_step)
        return layers
#----------------------------------------------------------
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

#G = createGraph(nb_nodes, nb_edges)

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

#chart1 = drawPlot1(G)
#col2.chart(chart1)
#col2.altair_chart(chart1, use_container_width=True)
#----------------------------------------------------------------------

# Add a additional row
# Rows with flight PTY-X
def addRow(df, new_od):
    last_row = df.tail(1).copy()
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

#--------------------------------------------------------------------------

# It will be more efficient to preprocess the fsu table (for one day or the entire table). 
# For each fight reaching a city other than PTY, compute the connecting flight (the same tail). 
# Note that some days, there are multiple tails. 
# Perhaps start with a groupby by tail? 

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

# Tracking by tail works most of the time, but not all the time. 
# I need a better algorithm to find connections at outside cities (non-PTY)

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
    #st.write("flight_pairs: ", flight_pairs)
    #st.write("bad_pairs: ", bad_flight_pairs)

    # Apparently, 
    # Tail HP1722 was transferred from SJO to TGU on 2019/10/01
    # Tail HP1829: not clear. But something happened on 2019/10/01
    # Tail HP1857 was transferred from TGU to SJO on 2019/10/01
    tail1722 = fsu[fsu['TAIL'] == 'HP1722']
    tail1829 = fsu[fsu['TAIL'] == 'HP1829']
    tail1857 = fsu[fsu['TAIL'] == 'HP1857']

    #st.write("tail1722: ", tail1722[cols])
    #st.write("tail1829: ", tail1829[cols])
    #st.write("tail1857: ", tail1857[cols])

    # A flight arrives at SJO at 14:16 pm (Zulu), HP1722. 
    # Two different nearest flights depart SJO at 14:25 and 17:28. 
    # They are different tails. So their departures can only be affected
    # by incoming pax. 
    fsu_sjo = fsu[fsu['OD'] == 'SJOPTY']
    #st.write("fsu_sjo: ", fsu_sjo[cols].sort_values('SCH_DEP_TMZ'))
    return flight_pairs, bad_flight_pairs

#-------------------------------------------------------------

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
            #st.write("<<<< fsu_X.shape[0]: ", fsu_X.shape[0])
            return pd.DataFrame()

        fsu_X = fsu_X.sort_values('SCH_DEP_TMZ')
        return fsu_X

    #-------------------------
    def findNextDepartures(f_od, pty_outbound_id, pty_inbounds):
        """
        Return the two departures closest to the arrival time of pty_outbound_id
        Return two full records (or only one if only one is available
        Return an empty frame if there is no next flight on that day.
        """
        arr_time = pty_outbound_id[16:20]
        lst = pty_inbounds[pty_inbounds['SCH_DEP_TMZ'] > arr_time]

        if lst.shape[0] == 0:
            return pd.DataFrame()

        if lst.shape[0] <= 2:
            return lst
        else:
            return lst.iloc[0:2]

        #lst = findNextDepartures(df, pty_outbound_id, pty_inbounds)
        #st.write("lst: ", lst[cols])
        #return lst

    #-------------------------
    def inboundOutboundDict(fsu, f_od, outbound_ids):
        """
        Given a list of outbound id's [fsu_ids], compute the corresponding inbound flights. The inbound flights are stored in a dataframe of one or two rows corresponding to the next or two next departures. 

        Arguments: 
          fsu: Flight properties (based on the FSU table), inbound and outbound
          f_od (DataFrame): List of inbound flights sorted first by OD, then by SCH_DEP_TMZ
          outbound_ids (list): list of outbound identifiers

        Return:
          dct: a dictionary with key (outbound id), and item (the next 1 or 2 departing flights) stored in a dataframe.
        """

        if False:
            st.write(">> fsu columns: ", fsu.columns)
            st.write("f_od columns: ", f_od.columns)
            st.write("fsu_ids columns: ", outbound_ids)
            st.write("f_od: ", f_od)

        dct = {}
        for outbound_id in outbound_ids: 
            pty_inbounds = getInbounds(fsu, outbound_id)
            if pty_inbounds.shape[0] == 0:
                # One flight has no inbounds on 2019/10/01
                #st.write("<<< outbound_id: ", outbound_id, ",  no inbounds")
                continue
            # 1 or 2 next departures
            next_departures = findNextDepartures(f_od, outbound_id, pty_inbounds)
            dct[outbound_id] = next_departures
        return dct

    # Precalculate (for one day) the 1-2 flights from city X following the inbound flight. The assumption is that one of these will be the correct flight to follow. Ideally, the tail should be the same as the incoming flight. 

    # Given an OD pair PTY-X, list all departures in ascending order, Zulu.
    f_od = fsu[cols].sort_values(['OD','SCH_DEP_TMZ'])

    fsu_ids = f_od[f_od['OD'].str[0:3] == 'PTY']['id'].values

    if False:
        st.write("f_od: ", f_od)
        st.write("=========================")
        st.write("fsu_ids.shape: ", fsu_ids.shape)

    dct = inboundOutboundDict(fsu, f_od, fsu_ids)
    return dct

#-------------------------------------------------------------------
def misc1(fsu):
    """
    """
    fsu = groupFSUbyTail(fsu)
    fsu = fsu.reset_index(drop=True)
    #st.write("fsu: ", fsu['TAIL'])

    # Ids of outbound flights originating in PTY 
    indices  = fsu.index[fsu['OD'].str[0:3] == 'PTY']
    #st.write(indices)

    fsu_outbound = fsu.iloc[indices]
    fsu_inbound  = fsu.iloc[indices+1,:]   #<<< ERROR

    st.write("fsu.shape: ", fsu.shape)
    st.write("fsu_inbound.shape: ",  fsu_inbound.shape)
    st.write("fsu_outbound.shape: ", fsu_outbound.shape)
    
    #st.write("fsu: ",  fsu[['TAIL','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']])
    #st.write("fsu_outbound: ", fsu1[['TAIL','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']])
    #st.write("fsu_inbound: ",  fsu2[['TAIL','OD','SCH_DEP_TMZ','SCH_ARR_TMZ']])

    # What is fsu1 and fsu2

    return fsu_outbound, fsu_inbound

#---------------------------------------------------------------------
# Dictionary only computed once
def computeDict(fsu1, fsu2):
    dct = {}
    nodct = {}

    ids_PTYX = fsu1[['id','TAIL']]
    ids_XPTY = fsu2[['id','TAIL']]
    #st.write("ids_PTYX= ", ids_PTYX)
    #st.write("ids_XPTY= ", ids_XPTY)

    ptyx = ids_PTYX['id'].tolist()
    xpty = ids_XPTY['id'].tolist()
    
    for i in range(ids_PTYX.shape[0]):
        if ptyx[i][13:16] == xpty[i][10:13]:
            dct[ptyx[i]] = xpty[i]
        else: 
            nodct[ptyx[i]] = xpty[i]

    #st.write("dct items")
    #st.write(dct)
    #st.write("nodct items")
    #st.write(nodct)
    #st.write("dict length: ", len(dct))
    #st.write("nodict length: ", len(nodct))
    return dct, nodct

#--------------------------------------------------------------------
def misc_computations(fsu):
    """
    I forgot what these were for
    """
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

#---------------------------------------------------------
def restrictNumberOfNodes(node_df, edge_df):
    # Streamlit does not update properly when rerun, when changing nb
    # Bothersome
    #nb = 12  # 
    """
    8	6.5	320.8
    9	12.5	320.8
    10	18.5	320.8
    11	30.5	320.8
    12	0.5	480.8
    """
    #nb = 13  # Looks ok (sometimes), and sometimes I get all zeros. HOW? WhY?
             # Works when I remove all nodes not used. So the graph is connected. 
             # I get a "should not happen" in walker.py. 
             ## DRAW THE NETWORK. WHAT IS WRONG? Perhaps not a tree? 
             # There is probably an error in handle*Id(), which should be simplified. 
    """
    10	18.5	320.8
    11	30.5	320.8
    12	0.5	480.8   # level 2
    13	0	0    # THIS IS AN ERROR
    """#
    #nb = 14  # There is an x coordinate of zero for node 13 
    """
    12	0.5	480.8
    13	0	0
    14	6.5	480.8
    """

    # Node 13 does not appear to be in the edge_df. Did I make an error? 
    # How can node 13 appear if there is no edge? 
    #xx = edge_df[(edge_df['id_f_y'] == 13) or (edge_df['id_nf_y'] == 13)]
    #xx = edge_df[edge_df['id_f_y'] == 13]
    #st.write("edges with 13 as a node: ", xx)
    #xx = edge_df[edge_df['id_nf_y'] == 13]
    #st.write("edges with 13 as a node: ", xx)
    # There are no edges with node 13 as a node. This indcates an error. 
    # I must go back to hande*Id() and find out where I made the mistake. 
    #st.stop() # <<<<

    # UNCOMMENT to limit the number of nodes
    #node_df = node_df.iloc[0:nb+1]
    #edge_df = edge_df[edge_df['id_f_y'] <= nb]
    #edge_df = edge_df[edge_df['id_nf_y'] <= nb]
    #st.write(".node_df: ", node_df)
    #st.write(".edge_df: ", edge_df)
    #st.write(f'gordon, {nb}')

def transform_for_walker(node_df, edge_df):
    ids = node_df['id'].tolist()
    nodes = {}

    w = Walker(debug=True, rootX=0.5, rootY=0.8)
    w.config['NODE_SIZE'] = 2
    w.config['NODE_SEPARATION'] = 4
    w.config['TREE_SEPARATION'] = 4

    for fid in ids:
        nodes[fid] = Node(ID=fid)
        w.add_node(nodes[fid])

    #for node in w.nodes:
        #st.write("Walker node id list: ", node.id)

    #for key, val in nodes.items():
        #st.write("Walker node id list: ", key, val, val.id)
        #st.write("++++++")

    st.write(edge_df.columns)
    src = edge_df['id_f_y']
    dest = edge_df['id_nf_y']

    st.write(node_df)
    st.write(edge_df)

    for s, d in zip(src, dest):
        nodes[d].parent = nodes[s]
        nodes[s].children.append(nodes[d])

    for node in w.nodes:
        children = node.children
        for i in range(len(node.children)):
            try:
                #st.write("i= ", i, "... len(children) 1: ", len(children))
                node.children[i].right_sibling = node.children[i+1]
                #st.write("try 1")
            except:
                #st.write("except 1")
                pass
                #st.write("i= ", i, "... len(children) 1: ", len(children))
            if i > 0:
                node.children[i].left_sibling =node.children[i-1]
                #st.write("try 2")

    for i, node in enumerate(w.nodes):
        #st.write("=======================")
        #st.write("i= ", i," ...node id: ", node.id)
        #st.write("len(children): ", len(node.children))
        try:
            #st.write("node left sibling: ", node.left_sibling.id)
            pass
        except:
            pass
        try:
            #st.write("node right sibling: ", node.right_sibling.id)
            pass
        except:
            pass
        try:
            #st.write("node parent: ", node.parent.id)
            pass
        except:
            pass
    return w, nodes

#-----------------------------------------------------------
def removeAdditionalEdges(node_df, edge_df):
    # Compute the in-degree of all nodes. Nodes with in-degree greater
    # than unity havfe multiple parents. 
    in_degree = defaultdict(int)  # in_degree for each node
    avail_d = defaultdict(list)  # in_degree for each node

    src = edge_df['id_f_y']
    dest = edge_df['id_nf_y']
    avail = edge_df['avail']
    edge_ids = edge_df['id_f_nf']
    removed_edges = []

    for s, d, avail, edge_id in zip(src, dest, avail, edge_ids):
        in_degree[d] += 1
        avail_d[d].append((avail, edge_id))

    for k,v in in_degree.items():
        if v > 1:
            # identify the edge with smallest avail_d. Keep it and 
            # remove the others. 
            avail_d[k].sort(key=lambda x: x[0])
            # edge to keep: 
            keep = avail_d[k][0][1]
            remove =  [avail_d[k][i][1] for i in range(1,len(avail_d[k]))]
            removed_edges.extend(remove)

    # Identify the edges connected to nodes with in_degree > 1. HOW?
    # return edges_to_remove, edges_removed
    #st.write("removed_edges: ", removed_edges)
    #st.write("edge_df: ", edge_df.shape)
    #st.write("edge_df: ", edge_df)
    edge_df = edge_df.set_index('id_f_nf', drop=True).drop(index=removed_edges, axis=0).reset_index()

    #st.write("edge_df: ", edge_df.shape)
    #st.write("edge_df: ", edge_df)

    # Should have saved the entire row instead of the removed edge ids
    return edge_df, removed_edges

#---------------------------------------------------------------
def convertIdsToInts(node_df, edge_df):
    str2id = {}
    for i in range(node_df.shape[0]):
        fid = node_df.iloc[i].id
        str2id[fid] = i
    fids = list(range(0,node_df.shape[0]))
    node_df.id = fids   # Do not relabel

    id_f_y = edge_df['id_f_y'].values
    id_nf_y = edge_df['id_nf_y'].values
    for i in range(edge_df.shape[0]):
        id_f_y[i] = str2id[id_f_y[i]]
        id_nf_y[i] = str2id[id_nf_y[i]]
    edge_df.id_f_y = id_f_y  # do not relabel
    edge_df.id_nf_y = id_nf_y  # do not relabel

    return node_df, edge_df


#-------------------------------------------------------------
def drawPlot3(node_df, edge_df, edge_structure, which_tooltip, rect_color, text_color):

    nb_nodes = node_df.shape[0]
    nb_edges = edge_df.shape[0]

    # Remove duplicates in nb_nodes: 2019/10/01PTYROS20:16805 is duplicate
    node_df = node_df.drop_duplicates()

    # THE DUPLICATES MUST BE REMOVED in handle*Id() before edge array is filled

    node_df = node_df.reset_index(drop=True)
    edge_df = edge_df.reset_index(drop=True)

    # CUNPTY, id: 2019/10/01/CUNPTY12:50354 appears twice! HOW!
    # I must make sure the tails match

    # transform all ids into integers. Create dictionary str_id ==> int_id
    # no concern for efficiency since this is for debugging Walker's algorithm

    node_df, edge_df = convertIdsToInts(node_df, edge_df)

    st.write("node_df: ", node_df)
    st.write("edge_df: ", edge_df)

    # For debugging
    # node_df, edge_df = restrictNumberOfNodes(node_df, edge_df)

    # The problem that at 3 levels, I no longer have a tree. Certain 
    # outbound flights have multiple feeders. That obviously leads to 
    # complexities. To transform the graph to a tree, various edges must
    # be removed. For starters, we will select the feeder with the minimum
    # connection time, since it is the most important. Once the graph 
    # is computed, the edges will be reinserted. 

    edge_df1, removed_edges = removeAdditionalEdges(node_df, edge_df)
    w, nodes = transform_for_walker(node_df, edge_df1)
    #st.write("shapes: ", edge_df.shape, edge_df1.shape)

    w.position_tree()
    # I got the export, but not the correct positioning
    w.export_to_frame('w_node_df', 'w_edge_df')

    #st.write("edge_structure: ", edge_structure)
    if edge_structure == 'Tree':
        edge_df = edge_df1.copy()


    w_nodes_df = pd.read_csv("w_node_df")

    # merge with w_nodes to incorporate (x,y) into node_df
    # cannot use concat since the node orders might differ
    node_df = pd.merge(node_df, w_nodes_df, how='inner', on='id')

    # Compute extent of domain
    xmin = node_df.x.min()
    xmax = node_df.x.max()
    ymin = node_df.y.min()
    ymax = node_df.y.max() 
    xmin -= 0.05 * (xmax - xmin)
    xmax += 0.05 * (xmax - xmin)
    ymin -= 0.15 * (ymax - ymin)
    ymax += 0.15 * (ymax - ymin)
    #st.write("min/max: ", xmin, xmax)
    #st.write("min/max: ", ymin, ymax)

    #----------------------

    xscale = alt.Scale(domain=[xmin, xmax])
    yscale = alt.Scale(domain=[0.,ymax])

    # Create and draw edges as a series of horizontal and vertical lines
    #df_step = createStepLines(node_df, edge_df)

    # SAVE TO FILE
    node_df.to_csv("node_df.csv", index=0)
    edge_df.to_csv("edge_df.csv", index=0)


    #layers = drawStepEdges(df_step, scale=xscale)

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
        width=20,
        height=20,
        opacity=1.0,
        color=rect_color,
        align='center',
    ).encode(
        x = alt.X('x:Q', scale=xscale),
        y = 'y:Q',
        #color = 'arr_delay', # not drawn if NaN
    )

    node_tooltips = alt.Chart(node_df).mark_circle(
        size=500,
        opacity=0.0,
    ).encode(
        #x = 'x:Q',
        x = alt.X('x:Q', scale=xscale),
        y = 'y:Q',
        tooltip=['id','SCH_DEP_TMZ','SCH_ARR_TMZ','arr_delay','dep_delay','od','FLT_NUM','TAIL','x','y']
    )

    node_text = alt.Chart(node_df).mark_text(
        opacity=1.,
        color=text_color,
        align='center',
        baseline='middle'
    ).encode(
        x = alt.X('x:Q', scale=xscale),
        y = 'y:Q',
        text='od',
        size=alt.value(10)
    )

    #st.write("Draw edges, edge_df: ", edge_df)
    # Create a data frame with as many columns as there are edges. 
    # Each column is four points. 
    edges = alt.Chart(edge_df).mark_rule(   # all edges
        strokeOpacity=1.0,
        stroke='yellow',
        strokeWidth=1.,
    ).encode(
        x = alt.X('x:Q', scale=xscale),
        y = 'y:Q',
        x2 = 'x2:Q',
        y2 = 'y2:Q',
        #stroke = 'pax:Q',
        #strokeWidth = 'pax:Q' #'scaled_pax:Q'
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

    mid_edges = alt.Chart(edge_df).mark_circle(color='yellow', size=200, opacity=0.8).encode(
        x = alt.X('mid_x:Q', scale=xscale),
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
        mid_edges = mid_edges.add_selection(
            edge_nearest
        )
    elif which_tooltip == 'Node':
        node_tooltips = node_tooltips.add_selection(
            node_nearest
        )
    elif which_tooltip == 'Off':
        pass

    #full_chart = (layers + nodes) # + node_text + node_tooltips + mid_edges)
    #full_chart = (layers + node_text) # + edges + nodes + node_text + node_tooltips + mid_edges)
    #full_chart = (layers + edges + nodes + node_text + node_tooltips + mid_edges)
    full_chart = (edges + nodes + node_text + node_tooltips + mid_edges)
    #full_chart = (nodes + node_tooltips)

    # Chart Configuration
    full_chart = full_chart.configure_axisX(
        labels=True,
    )

    return full_chart

#----------------------------------------------------------------
def computePos(node_df, edge_df, xmax, ymax, dx, dy):
    """ 
    Compute positions of nodes encoded in the dataframe  node_df
    """
    # By convention, the nodes with even levels are feeders (0,2,4)

    nb_nodes = node_df.shape[0]

    levels = node_df['lev'].value_counts().to_frame()

    # Compute deltax per level

    W = xmax
    deltax = {}
    deltay = {}
    for lev in range(levels.shape[0]):
        deltax[lev] = W / levels['lev'].values[lev]
        deltay[lev] = 0.3

    # Each parent city should have a different delta y
    # higher levels should have higher delta y (perhaps logarithmic scaling?)
    # Draw edge with larger delays in red, or thicker
    node_df['y'] = 0.8 - node_df['lev'] * deltay[0]

    # Compute number of children of each node
    nb_children = defaultdict(int)
    parent = defaultdict(int)
    # specify the parent, get a list of children
    children = defaultdict(list)

    edge_df = edge_df.reset_index(drop=True)  # To be safe
    for i in range(edge_df.shape[0]):
        src_id = edge_df.loc[i,'id_f_y']
        target_id = edge_df.loc[i,'id_nf_y']
        parent[target_id] = src_id
        children[src_id].append(target_id)
        nb_children[src_id] += 1

    node_df = node_df.set_index('id')
    ids = list(node_df.index)

    #st.write("len(children): ", len(children))

    # I want to start at the root 
    # assuming parents are stored in random order, how to fill the x values? 
    # This is where NetworkX would be very useful
    #

    root_id = node_df.index[0]
    par = root_id
    node_df.loc[par,'x'] = 0.05
    node_df.loc[par,'y'] = 0.8

    stack = deque()
    stack.append(par)

    # Keep this non-recursive. 
    # Use a stack of parents to process
    while True:
        # Assign x,y to all children
        try:
            par = stack.pop()
        except:
            break
        v = children[par]
        nb_children = len(v)
        if nb_children == 1:
            node_df.loc[v[0],'x'] = node_df.loc[par,'x']
            node_df.loc[v[0],'y'] = node_df.loc[par,'y'] - dy
            stack.append(v[0])
        elif nb_children > 1:
            for i,c in enumerate(v):
                node_df.loc[v[i],'y'] = node_df.loc[par,'y'] - dy
                node_df.loc[v[i],'x'] = node_df.loc[par,'x'] + i*dx
                stack.append(v[i])

    node_df = node_df.reset_index()
    st.write("nodes with x,y: ", node_df)
    return node_df

#----------------------------------------------------------------------
