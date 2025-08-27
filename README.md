### 深さ優先探索
```python
import networkx as nx
from collections import deque

def my_dfs(G, s, traversed):
    traversed.add(s)
    print(s)
    
    for v in G.neighbors(s):
        if v not in traversed:
            my_dfs(G, v, traversed)

# スタックを用いたver.
def dfs_stack(G, s):
    traversed = set()
    q = deque()
    q.appendleft(s)
    
    while q:
        u = q.popleft()
        if u not in traversed:
            print(u)
            traversed.add(u)
            
            for v in G.neighbors(u):
                if not v in traversed:
                    q.appendleft(v)
```



### オイラー閉路
無向連結グラフを入力として、そのグラフにオイラー閉路が存在するならTrueを、存在しないならFalseを返す関数
```python
import networkx as nx

def my_euler_test(G):
    for v in G.nodes():
        d = G.degree[v]
        if d % 2 == 1:
            return False
    return True
```



### Dijkstra
```python
import networkx as nx

def my_extract_min(D, X):
    arg_min = -1
    min_value = float('inf')
    
    for i in range(len(D)):
        if D[i] < min_value:
            if i in X:
                arg_min = i
                min_value = D[i]
    
    return arg_min

def my_Dijkstra(G, s):
    X = set(G.nodes)
    D = [float('inf')] * nx.number_of_nodes(G)
    D[s] = 0
    
    while X:
        u = my_extract_min(D, X)
        X.remove(u)
        for v in G.neighbors(u):
            if v in X:
                new_distance = D[u] + G.edges[u, v]['weight']
                if D[v] > new_distance:
                    D[v] = new_distance
    return D
```



### 2分ヒープを用いたDijkstra
```python
import networkx as nx
import heapq

def my_heap_Dijkstra(G, s):
    X = set(G.nodes)
    D = [float('inf')] * nx.number_of_nodes(G) 
    D[s] = 0
    Q = []
    heapq.heappush(Q, (D[s], s))
    while Q:
        u_distance, u = heapq.heappop(Q)
        if D[u] < u_distance:
            continue
        X.remove(u)
        for v in G.neighbors(u):
            if v in X:
                new_distance = u_distance + G.edges[u, v]['weight']
                if D[v] > new_distance:
                    D[v] = new_distance
                    heapq.heappush(Q, (new_distance, v))
    return D
```



### Prim
```python
import networkx as nx

def my_extract_min(D, X):
    arg_min = -1
    min_value = float('inf')
    
    for i in range(len(D)):
        if D[i] < min_value:
            if i in X:
                arg_min = i
                min_value = D[i]
    
    return arg_min

def my_Prim(G, s):
    X = set(G.nodes)
    D = [float('inf')] * nx.number_of_nodes(G)
    D[s] = 0
    P = [-1] * nx.number_of_nodes(G)
    A = []
    while X:
        u = my_extract_min(D, X)
        X.remove(u)
        if u != s:
            A.append((P[u], u))
        for v in G.neighbors(u):
            if v in X:
                new_distance = G.edges[u, v]['weight']
                if D[v] > new_distance:
                    D[v] = new_distance
                    P[v] = u
    return A
```



### Bellman-Ford
```python
import networkx as nx

def my_Bellman_Ford(G, s):
    n = nx.number_of_nodes(G)
    D = [float('inf')] * n
    D[s] = 0
    
    for i in range(1, n):
        D_new = D[:]
        
        for u, v in G.edges():
            new_distance = D[u] + G.edges[u, v]['weight']
            if D_new[v] > new_distance:
                D_new[v] = new_distance
        D = D_new
    
    for u, v in G.edges():
        if D[v] > D[u] + G.edges[u, v]['weight']:
            return (False, D)

    return (True, D)
```



### グラフGから行列Aを作成する関数
```python
import networkx as nx
import numpy as np

def transition_matrix(G): 
    n = nx.number_of_nodes(G)
    A = nx.to_numpy_array(G, nodelist=range(n), weight=None)
    A = A.transpose()
    cols_sum = A.sum(axis=0)
    
    for i in range(n):
        s = cols_sum[i]
        if s != 0:
            A[:,i] = A[:,i]/s
        else:
            A[:,i] = 1/n
    return A

# 更新操作を行う関数
def update(G, v, d, k):
    A = transition_matrix(G)
    n = nx.number_of_nodes(G)
    J = np.ones((n, n))/n
    Mg = d * A    + (1-d) * J
    
    for _ in range(k):
        v = Mg @ v
    
    return v
```



### HITSの更新操作を行う関数
```python
import networkx as nx
import numpy as np

def HITS_update(G, h0, k): 
    n = nx.number_of_nodes(G)
    A = nx.to_numpy_array(G, nodelist=range(n), weight=None)
    AT = A.transpose()
    
    hk = h0
    for _ in range(k):
        ak = AT @ hk
        hk = A @ ak
        
        ak = ak/ak.sum(axis=0)[0]
        hk = hk/hk.sum(axis=0)[0]
    return ak, hk
```



### Ford-Fulkerson
```python
import networkx as nx
from collections import deque

# 残余ネットワークから増加経路があるか判定する関数
def find_augmentpath(N, s, t):
    P = [-1] * nx.number_of_nodes(N)
    visited = set()
    stack = deque()
    stack.appendleft(s)
    while stack:
        v = stack.popleft()
        if v == t:
            return P, True
        if not v in visited:
            visited.add(v)
            for w in N.neighbors(v):
                if not w in visited and N.edges[v, w]['weight'] > 0:
                    stack.appendleft(w)
                    P[w] = v
    return P, False

def restore_shortestpath(u, v, P):
    path = []
    temp = v
    while temp != u:
        parent = P[temp]
        path.append((parent, temp))
        temp = parent
    path.reverse()
    return path


def min_capacity(N, path):
    min_cap = float('inf')
    for u, v in path:
        capacity = N.edges[u, v]['weight']
        if capacity < min_cap:
            min_cap = capacity
    return min_cap

def increase_flow(G, N, path, amount, flow):
    for u, v in path:
        if G.has_edge(u, v):
            flow[(u, v)] += amount
        else:
            flow[(v, u)] -= amount
        N.edges[u, v]['weight'] -= amount
        if N.has_edge(v, u):
            N.edges[v, u]['weight'] += amount
        else:
            N.add_edge(v, u, weight=amount)


def my_Ford_Fulkerson(G, s, t):
    N = G.copy()
    f = {}
    for u, v in N.edges:
        f[(u, v)] = 0
    P, is_found = find_augmentpath(N, s, t)
    while is_found:
        augmentpath = restore_shortestpath(s, t, P)
        min_cap = min_capacity(N, augmentpath)
        increase_flow(G, N, augmentpath, min_cap, f)
        P, is_found = find_augmentpath(N, s, t)
    return N, f
```



### 最小カット
Ford-Fulkersonアルゴリズムの関数を用いる
```python
import networkx as nx
from collections import deque

def mincut_dfs(N, s):
    visited = set()
    stack = deque()
    stack.appendleft(s)
    while stack:
        v = stack.popleft()
        if not v in visited:
            visited.add(v)
            for w in N.neighbors(v):
                if not w in visited and N.edges[v, w]['weight'] > 0:
                    stack.appendleft(w)
    return visited, set(N.nodes)-visited
```



### Needleman-Wunsch
```python
import numpy as np

def my_Needleman_Wunsch(s1, s2, M, d):
    m = len(s1)
    n = len(s2)
    D = np.zeros((m+1, n+1))
    D[0, 0] = 0
    for i in range(1, m+1):
        D[i, 0] = -d*i
    for j in range(1, n+1):
        D[0, j] = -d*j
    for i in range(1, m+1):
        for j in range(1, n+1):
            D[i, j] = max(D[i, j-1] - d, D[i-1, j] - d, D[i-1, j-1] + M[s1[i-1], s2[j-1]])
    return D[m, n]
```



### セル・オートマトン
```python
class Cellular_Automaton:
    def __init__(self, cells, ruleset):
        self.cells = cells.copy()
        self.len = len(cells)
        self.ruleset = ruleset
    def current_state(self):
        return self.cells
    def to_str(self, char):
        result = ''
        for i in range(self.len):
            if self.cells[i] == 0:
                result += ' '
            else:
                result += char
        return result
    def __str__(self):
        result = ''
        for i in range(self.len):
            if self.cells[i] == 0:
                result += ' '
            else:
                result += '*'
        return result
    def transition(self):
        next_cells = [0] * self.len
        for i in range(self.len):
            neighbors = (self.cells[(i-1) % self.len], self.cells[i], self.cells[(i+1) % self.len])
            next_cells[i] = self.ruleset[neighbors]
        self.cells = next_cells

# 実行例
ruleset = { (0, 0, 0) : 0, (0, 0, 1): 1, (0, 1, 0) : 1, (0, 1, 1) : 1, (1, 0, 0) : 1, (1, 0, 1) : 0, (1, 1, 0) : 0, (1, 1, 1) : 0}
init_state = [0] * 75
init_state[37] = 1
ca = Cellular_Automaton(init_state, ruleset)

for _ in range(37):
    print(ca)
    ca.transition()
```
