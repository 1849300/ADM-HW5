import pandas as pd
import time
import networkx as nx


def create_txt(f, start, stop, *args):
    ''' This function creates text files with only interactions that belong in the given time interval '''

    # If optional parameters are given -> we build trees based on two time intervals
    if args != ((),):
        # We convert in timestamp also the start and the end of the second interval
        start2 = time.mktime(time.strptime(args[0][0], '%Y-%m-%d'))
        stop2 = time.mktime(time.strptime(args[0][1], '%Y-%m-%d'))

    if f == 'a':
        # f = 'a' means that we have to build all trees and not only one (we made that to reuse this function also for functionality one)
        for f in ['a2q', 'c2a', 'c2q']:
            # Reads the file as dataframe (adding the column names)
            fn = 'sx-stackoverflow-' + f + '.txt'
            df = pd.read_csv(fn, sep=' ')
            df.columns = ['u', 'v', 'timestamp']
            # Filter dataframe
            cond_ = (df['timestamp'] >= start) & (df["timestamp"] <= stop)
            sub_data = df.loc[cond_, :]
            if args != ((),):
                # If we are dealing with two intervals so we add also the rows in the second interval
                cond_ = (df['timestamp'] >= start2) & (
                    df["timestamp"] <= stop2)
                sub_data2 = df.loc[cond_, :]
                sub_data = sub_data.append(sub_data2, ignore_index=True)
            # Write dataframe in a file
            sub_data.to_csv(f+'.txt', sep=',', index=False, header=False)
    else:
        # As above but with only one file (graph)
        fn = 'sx-stackoverflow-' + f + '.txt'
        df = pd.read_csv(fn, sep=' ')
        df.columns = ['u', 'v', 'timestamp']
        cond_ = (df['timestamp'] >= start) & (df["timestamp"] <= stop)
        sub_data = df.loc[cond_, :]
        if args != ((),):
            cond_ = (df['timestamp'] >= start2) & (df["timestamp"] <= stop2)
            sub_data2 = df.loc[cond_, :]
            sub_data = sub_data.append(sub_data2, ignore_index=True)
        sub_data.to_csv(f+'.txt', sep=',', index=False, header=False)


def build(f, dstart, dstop, *args):
    ''' Build trees starting from files '''

    # We convert in timestamp the start and end dates
    start = time.mktime(time.strptime(dstart, '%Y-%m-%d'))
    stop = time.mktime(time.strptime(dstop, '%Y-%m-%d'))

    create_txt(f, start, stop, args)

    G = nx.DiGraph()  # Initialize graph

    if f == 'a':  # If we want to create the graph based on all the 3 files
        w = 4
        for el in ['a2q', 'c2a', 'c2q']:
            # It halves the weight each time it opens a new file in order to give different
            # weights to different interactions between users
            w /= 2
            with open(el+'.txt') as f:
                # Reads file as text
                for line in f:
                    # Split text line and convert the content in integers
                    u, v, _ = map(int, line.split(','))
                    # We won't model "self-interactions"
                    if u != v:
                        if (u, v) in G.edges:
                            # If the edge is already in the graph we update its weight
                            tmp_w = G.get_edge_data(u, v)['weight']
                            G.add_edge(u, v, weight=w+tmp_w)
                        else:
                            # If not, we create a new edge
                            G.add_edge(u, v, weight=w)
    else:
        # These if set w (weight) with respect to graph (the one we will build)
        if f == 'a2q':
            w = 2
        elif f == 'c2a':
            w = 1
        elif f == 'c2q':
            w = 0.5
        with open(f+'.txt') as f:
            # As above
            for line in f:
                u, v, _ = map(int, line.split(','))
                if u != v:
                    if (u, v) in G.edges:
                        tmp_w = G.get_edge_data(u, v)['weight']
                        G.add_edge(u, v, weight=w+tmp_w)
                    else:
                        G.add_edge(u, v, weight=w)

    return G
