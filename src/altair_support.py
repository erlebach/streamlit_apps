from src.template import *

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

# Chart using Copa data

