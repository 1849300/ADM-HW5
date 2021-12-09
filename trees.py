import pandas as pd
import os
import time
import networkx as nx

# def check():
#     ''' It returns true if at least one csv is missing'''

#     return "a2q.txt" not in os.listdir('./') or "c2a.txt" not in os.listdir('./') or "c2q.txt" not in os.listdir('./')


def create_txt(f, start, stop):
    if f == 'a':
        for f in ['a2q', 'c2a', 'c2q']:
            fn = 'sx-stackoverflow-' + f + '.txt'
            df = pd.read_csv(fn, sep=' ')
            df.columns = ['u', 'v', 'timestamp']
            cond_ = (df['timestamp'] >= start) & (df["timestamp"] <= stop)
            sub_data = df.loc[cond_, :]
    else:
        fn = 'sx-stackoverflow-' + f + '.txt'
        df = pd.read_csv(fn, sep=' ')
        df.columns = ['u', 'v', 'timestamp']
        cond_ = (df['timestamp'] >= start) & (df["timestamp"] <= stop)
        sub_data = df.loc[cond_, :]
    sub_data.to_csv(f+'.txt', sep=',', index=False, header=False)


def build(f, dstart, dstop):
    start = time.mktime(time.strptime(dstart, '%Y-%m-%d'))
    stop = time.mktime(time.strptime(dstop, '%Y-%m-%d'))

    # if check():  # If there aren't txt files it calls the function to create them
    create_txt(f, start, stop)

    G = nx.DiGraph()  # Initialize graph

    if f == 'a':  # If we want to create the graph based on all the 3 files
        w = 4
        for el in ['a2q', 'c2a', 'c2q']:
            # filter_dataset(el, start, stop)
            w /= 2  # It halves the weight each time it opens a new file
            with open(el+'.txt') as f:
                for line in f:
                    u, v, _ = map(int, line.split(','))
                    if u != v:
                        if (u, v) in G.edges:
                            tmp_w = G.get_edge_data(u, v)['weight']
                            G.add_edge(u, v, weight=w+tmp_w)
                        else:
                            G.add_edge(u, v, weight=w)
    else:
        # filter_dataset(f, start, stop)
        if f == 'a2q':
            w = 2
        elif f == 'c2a':
            w = 1
        elif f == 'c2q':
            w = 0.5
        with open(f+'.txt') as f:
            for line in f:
                u, v, _ = map(int, line.split(','))
                if u != v:
                    if (u, v) in G.edges:
                        tmp_w = G.get_edge_data(u, v)['weight']
                        G.add_edge(u, v, weight=w+tmp_w)
                    else:
                        G.add_edge(u, v, weight=w)

    return G
