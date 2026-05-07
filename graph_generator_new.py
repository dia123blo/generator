import igraph as ig
import numpy as np
import argparse
import os
import math
import json
from collections import Counter, defaultdict
from utils_l2f import generate_labels, generate_tree, get_direction, powerset, sample_element, str2bool, retrieve_multiple_edges,generate_vf_random_in_range,generate_ef_random_in_range
from pattern_checker import PatternChecker
from pattern_generator import generate_patterns
from time import time

DECAY = 0.954 # 2 sigma

class NEC(object):
    def __init__(self, data=None, adj=None, inter_adj=None, vertex_label=None, \
                 feature=None, is_clique=False, nec_id=0):
        if data is None:
            self.data = list()
        else:
            try:
                iterator = iter(data)
                self.data = data
            except TypeError as e:
                self.data = [data]
        self.adj = adj
        self.inter_adj = inter_adj
        self.vertex_label = vertex_label
        self.feature = feature
        self.is_clique = False
        self.nec_id = nec_id
    
    def append(self, item):
        self.data.append(item)
    
    def extend(self, items):
        self.data.extend(items)

    def __len__(self):
        return self.data.__len__()
    
    def __setitem__(self, idx, v):
          self.data[idx] = v

    def __getitem__(self, idx):
          return self.data[idx]

class NECTree(object):
    def __init__(self, vcount, directed=True):
        self.tree = ig.Graph(directed=directed)
        self.NEC_by_adj = dict()
        self.NEC_by_vertex_label = dict()
        self.NEC_by_vertex_feature = dict()
        self.NEC_by_vertex_index = [None] * vcount
    
    def add_nec(self, nec):
        adj = nec.adj
        vertex_label = nec.vertex_label
        feature = nec.feature
        is_clique = nec.is_clique

        nec.nec_id = self.tree.vcount()
        
        self.tree.add_vertex(label=vertex_label, feature=feature)

        if adj not in self.NEC_by_adj:
            self.NEC_by_adj[adj] = list()
        self.NEC_by_adj[adj].append(nec)

        if vertex_label not in self.NEC_by_vertex_label:
            self.NEC_by_vertex_label[vertex_label] = list()
        self.NEC_by_vertex_label[vertex_label].append(nec)
        if feature not in self.NEC_by_vertex_feature:
            self.NEC_by_vertex_feature[feature] = list()
        self.NEC_by_vertex_feature[feature].append(nec)

        for vertex_index in nec:
            self.NEC_by_vertex_index[vertex_index] = nec


    def add_edge(self, source, target, edge_label, edge_feature):
        self.tree.add_edge(source, target, label=edge_label, feature=edge_feature)

class GraphGenerator(object):
    def __init__(self, pattern):
        self.pattern = pattern if pattern else ig.Graph(directed=True)
        self.number_of_pattern_vertices = self.pattern.vcount()
        self.number_of_pattern_edges = self.pattern.ecount()
        self.pattern_vertex_label_counter = Counter(self.pattern.vs["label"])
        self.pattern_edge_label_counter = Counter(self.pattern.es["label"])

        self.pattern_edge_label_mapping = defaultdict(set)
        self.pattern_edge_feature_mapping = defaultdict(set)
        self.pattern_vertex_edge_label_mapping = defaultdict(set)
        self.pattern_vertex_edge_feature_mapping = defaultdict(set)
        
        for edge in self.pattern.es:
            # print(f"Edge {edge.tuple}: Label={edge['label']}, Feature={edge['feature']}")
            self.pattern_edge_label_mapping[edge.tuple].add(edge["label"])
            self.pattern_edge_feature_mapping[edge.tuple].add(edge["feature"])
            key = (self.pattern.vs[edge.source]["label"], self.pattern.vs[edge.target]["label"])
            self.pattern_vertex_edge_label_mapping[key].add(edge["label"])
            self.pattern_vertex_edge_feature_mapping[key].add(edge["feature"])

        self.number_of_pattern_vertex_labels = int(max(self.pattern.vs["label"])) + 1
        self.number_of_pattern_edge_labels = int(max(self.pattern.es["label"])) + 1

        self.pattern_nec_tree = self.rewrite_to_nec_tree()
        self.pattern_nec_tree_vertex_edge_label_mapping = defaultdict(set)
        self.pattern_nec_tree_vertex_edge_feature_mapping = defaultdict(set)
        for edge in self.pattern_nec_tree.tree.es:
            # print(f"Edge {edge.tuple}:")
            # print(f"Label={edge['label']},")
            # print(f"Feature={edge['feature']}")
            key = (self.pattern_nec_tree.tree.vs[edge.source]["label"], self.pattern_nec_tree.tree.vs[edge.target]["label"])
            
            self.pattern_nec_tree_vertex_edge_label_mapping[key].add(edge["label"])
            self.pattern_nec_tree_vertex_edge_feature_mapping[key].add(edge["feature"])

        self.pattern_checker = PatternChecker()

    def choose_start_q_vertex(self):
        vs = list()  # 初始化候选顶点列表
        for v in self.pattern.vs:  # 遍历模式图所有顶点
            # 获取当前顶点标签在模式图中的出现频率
            freq = self.pattern_vertex_label_counter[v["label"]]
            
            # 备用方案：获取标签在整个目标图中的频率（当前被注释）
            # freq = self.graph_vertex_label_counter[v["label"]] 
            
            deg = v.degree()  # 计算当前顶点度数（邻居数量）
            vs.append((freq/deg, v.index))  # 存储(评分, 顶点索引)元组
        
        vs.sort()  # 按评分升序排序
        return vs[0][1]  # 返回评分最低的顶点索引
    
    def find_cliques(self, edges):
        # 步骤1: 收集所有独特顶点
        vs = set()
        for edge in edges:
            vs.update(edge)  # 添加边的两个端点
            
        # 步骤2: 排除单点情况
        if len(vs) < 2:
            return []  # 团至少需要2个顶点

        cliques = []
        # 步骤3: 遍历所有可能的顶点组合（至少2个顶点）
        for clique_vs in powerset(vs, min_size=2):
            n = len(clique_vs)
            
            # 步骤4: 统计子图内顶点的入度/出度
            in_degrees = Counter()
            out_degrees = Counter()
            for edge in edges:
                if edge[0] in clique_vs and edge[1] in clique_vs:
                    # 有向边：src→dst
                    out_degrees[edge[0]] += 1  # 源头出度+1
                    in_degrees[edge[1]] += 1    # 目标入度+1
            
            # 步骤5: 验证度序列
            in_d = in_degrees[clique_vs[0]]   # 首个顶点入度
            out_d = out_degrees[clique_vs[0]] # 首个顶点出度
            
            # 必要条件1: 非孤立顶点
            if in_d == 0:
                continue
                
            # 必要条件2: 入度需能被(n-1)整除（证明全连接）
            if in_d % (n-1) != 0:
                continue
                
            # 必要条件3: 入度=出度
            if in_d != out_d:
                continue
                
            # 必要条件4: 所有顶点入度/出度相同
            if not (all(in_d == in_degrees[v] for v in clique_vs) and
                    all(out_d == out_degrees[v] for v in clique_vs)):
                continue
                
            cliques.append(clique_vs)  # 验证通过，加入团列表
            
        return cliques

    def find_necs(self, group):
        # each vertex in the same group has the same label
        # so we do not need to care the vertex label here
        groups_by_adj = defaultdict(list)
        for v in group:
            adj = list()  # 存储(方向, 边标签, 邻接顶点)
            
            # 收集所有出边
            for out_e in self.pattern.incident(v, mode=ig.OUT):
                edge = self.pattern.es[out_e]
                u = edge.target
                adj.append((ig.OUT, edge["label"], u))
            
            # 收集所有入边
            for in_e in self.pattern.incident(v, mode=ig.IN):
                edge = self.pattern.es[in_e]
                u = edge.source
                adj.append((ig.IN, edge["label"], u))
            
            # 排序确保邻接结构可比性
            adj = tuple(sorted(adj))
            groups_by_adj[adj].append(v)  # 按邻接结构分组
        
        
            # NECs with same adj
            singleton_group_mapping = dict()  # 单顶点->其邻接结构
            necs = list()  # 结果NEC列表

            # 处理成组的顶点
            for adj, vs in groups_by_adj.items():
                if len(vs) > 1:
                    # 直接形成NEC（非团）
                    necs.append(NEC(sorted(vs), adj=adj, is_clique=False))
                else:
                    # 单顶点暂存后续处理
                    singleton_group_mapping[vs[0]] = adj
        
        # NECs with cliques and same adj-N_q
        # firstly check the indegree and outdegree
        groups_by_degree = defaultdict(list)
        for v, adj in singleton_group_mapping.items():
            # 计算入度和出度
            in_degree = len([x[0] == ig.IN for x in adj])
            out_degree = len(adj) - in_degree
            groups_by_degree[(in_degree, out_degree)].append(v)
            
        for key, vs in groups_by_degree.items():
            if len(vs) == 1:
                necs.append(NEC(vs, adj=singleton_group_mapping[vs[0]], is_clique=False))
            else:
                # check whether to form a clique
                inter_edges_by_edge_labels = defaultdict(set) # key: e_label, value: edges
                edge_label_set = set()
                for v in vs:
                    for x in singleton_group_mapping[v]:
                        if x[2] in vs:
                            if x[0] == ig.OUT:
                                src_tgt = (v, x[2])
                            else:
                                src_tgt = (x[2], v)
                            inter_edges_by_edge_labels[x[1]].add(src_tgt)
                
                cliques_by_edge_labels = dict() # key: e_label, value: cliques
                for edge_label, edges in inter_edges_by_edge_labels.items():
                    cliques = self.find_cliques(edges)
                    if len(cliques) > 0:
                        cliques_by_edge_labels[edge_label] = set(cliques)
                if len(cliques_by_edge_labels) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                        
                # find mixed cliques
                # if a clique with edge_label A appears in cliques with edge_label B, it is valid
                # A: (0,1), (1,2), (0,2), (0,1,2)
                # B: (0,1), (1,2), (0,2), (0,1,2)
                # result: (0,1), (1,2), (0,2), (0,1,2)
                
                # if a clique with edge_label A does not appear in cliques with edge_label B, it is invalid
                # A: (0,1), (1,2), (0,2), (0,1,2)
                # B: empty
                # result: empty
                mixed_cliques = set.intersection(*cliques_by_edge_labels.values())
                if len(mixed_cliques) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                
                # check the same outer adj
                valid_cliques = dict() # key: clique, value: (outer_adj, inter_adj)
                for clique in mixed_cliques:
                    # get inter_adj and outer_adj
                    inter_adjs = dict() # key: v_id, value: inter_adj
                    outer_adjs = dict() # key: v_id, value: outer_adj
                    for v in clique:
                        adj = singleton_group_mapping[v]
                        inter_adjs[v] = sorted([(x[0], x[1]) for x in adj if x[2] in clique]) # x[2] is useless because it is a clique
                        outer_adjs[v] = sorted([x for x in adj if x[2] not in clique])
                    
                    # check same outer adj
                    o_adj = next(iter(outer_adjs.values()))
                    if not all([o_adj == outer_adj for outer_adj in outer_adjs.values()]):
                        continue    
                    i_adj = next(iter(inter_adjs.values()))
                    valid_cliques[clique] = (o_adj, i_adj)
                if len(valid_cliques) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                
                # choose the larger cliques and remove subcliques
                # valid_cliques: (0,1), (1,2), (0,2), (0,1,2), (3,4,5)
                # result: (0,1,2), (3,4,5)
                final_cliques = list()
                for valid_clique in sorted(valid_cliques.keys(), key=lambda x: (-len(x), x)):
                    is_subclique = False
                    for final_clique in final_cliques:
                        if final_clique.issuperset(valid_clique):
                            is_subclique = True
                            break
                    if not is_subclique:
                        final_cliques.append(set(valid_clique))
                
                # merge vertices in one final cliques
                for final_clique in final_cliques:
                    final_clique = sorted(final_clique)
                    outer_adj, inter_adj = valid_cliques[tuple(final_clique)]
                    necs.append(NEC(final_clique, adj=tuple(outer_adj), inter_adj=tuple(inter_adj), is_clique=True))
                
                # add the left singleton NECs
                for v in set(vs).difference(set.union(*final_cliques)):
                    necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
        return necs

    def rewrite_to_nec_tree(self):
        nec_tree = NECTree(self.number_of_pattern_vertices, directed=True)

        start_v = self.choose_start_q_vertex()
        visited = [0] * self.number_of_pattern_vertices
        visited[start_v] = 1
        adj = list() # [(mode, e_label, v_id), ...]
        out_edges = retrieve_multiple_edges(self.pattern, source=start_v)
        in_edges = retrieve_multiple_edges(self.pattern, target=start_v)
        adj.extend([(ig.OUT, edge["label"], edge.target) for edge in out_edges])
        adj.extend([(ig.IN, edge["label"], edge.source) for edge in in_edges])
        adj = tuple(sorted(adj))
        
        root = NEC(data=[start_v], adj=adj, vertex_label=self.pattern.vs[start_v]["label"],\
                    feature=self.pattern.vs[start_v]["feature"],is_clique=False, nec_id=0)
        nec_tree.add_nec(root)

        v_current = list()
        v_next = [root]

        while len(v_next) > 0:
            v_current, v_next = v_next, list()
            for nec in v_current:
                groups = defaultdict(list) # key: (mode, edge_label, vertex_label), value: [v_id, ...]
                for v in nec:
                    # group by (edge_mode, edge_label, vertex_label)
                    out_edges = retrieve_multiple_edges(self.pattern, source=v)
                    in_edges = retrieve_multiple_edges(self.pattern, target=v)
                    for edge in sorted(out_edges, key=lambda x: x["label"]):
                        u = edge.target
                        if not visited[u]:
                            key = (ig.OUT, edge["label"], self.pattern.vs[u]["label"], self.pattern.vs[u]["feature"])
                            groups[key].append(u)
                            visited[u] = 1
                    for edge in sorted(in_edges, key=lambda x: x["label"]):
                        u = edge.source
                        if not visited[u]:
                            key = (ig.IN, edge["label"], self.pattern.vs[u]["label"], self.pattern.vs[u]["feature"])
                            groups[key].append(u)
                            visited[u] = 1
                for key, group in groups.items():
                    mode, edge_label, vertex_label, feature = key
                    new_necs = self.find_necs(group)
                    for new_nec in new_necs:
                        new_nec.vertex_label = vertex_label
                        new_nec.feature = feature
                        nec_tree.add_nec(new_nec)
                        if mode == ig.OUT:
                            nec_tree.add_edge(nec.nec_id, new_nec.nec_id, edge_label=edge_label, edge_feature=feature)
                        else:
                            nec_tree.add_edge(new_nec.nec_id, nec.nec_id, edge_label=edge_label, edge_feature=feature)
                        v_next.append(new_nec)
            v_next.sort(key=lambda x: x.vertex_label)
        return nec_tree

    def update_subgraphs(self, subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping):
        new_edges_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_keys_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_labels_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_features_in_subgraphs = [list() for i in range(len(subgraphs))]
        subgraphs_vlabels = [subgraph.vs["label"] for subgraph in subgraphs]
        subgraphs_vfeatures = [subgraph.vs["feature"] for subgraph in subgraphs]
        
        for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
            if sg1 == sg2:
                src_tgt = (v1, v2)
                key = (subgraphs_vlabels[sg1][v1], subgraphs_vlabels[sg2][v2])
                pattern_edge_labels = self.pattern_vertex_edge_label_mapping[key]
                # 只添加模式图中存在的边标签
                edge_labels_to_add = [edge_label for edge_label in edge_labels if edge_label in pattern_edge_labels]
                edge_features_to_add = [edge_feature for edge_label, edge_feature in 
                                      zip(edge_labels, graph_edge_feature_mapping[(sg1, v1, sg2, v2)]) 
                                      if edge_label in pattern_edge_labels]
                
                for i, (edge_label, edge_feature) in enumerate(zip(edge_labels_to_add, edge_features_to_add)):
                    new_edges_in_subgraphs[sg1].append(src_tgt)
                    new_edge_keys_in_subgraphs[sg1].append(i)
                    new_edge_labels_in_subgraphs[sg1].append(edge_label)
                    new_edge_features_in_subgraphs[sg1].append(edge_feature)
                    
        for sg, subgraph in enumerate(subgraphs):
            # 清除可能存在的重复边
            subgraph.delete_edges(None)
            subgraph.add_edges(new_edges_in_subgraphs[sg])
            subgraph.es["label"] = new_edge_labels_in_subgraphs[sg]
            subgraph.es["key"] = new_edge_keys_in_subgraphs[sg]
            subgraph.es["feature"] = new_edge_features_in_subgraphs[sg]



    def merge_subgraphs(self, subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping):
        graph = ig.Graph(directed=True)
        graph_vertex_mapping = list()
        graph_vertex_mapping_reversed = dict()
        for sg, subgraph in enumerate(subgraphs):
            for v_id in range(subgraph.vcount()):
                graph_vertex_mapping.append((sg, v_id))

        np.random.shuffle(graph_vertex_mapping)

        for v_id, x in enumerate(graph_vertex_mapping):
            graph_vertex_mapping_reversed[x] = v_id
            graph.add_vertex(label=subgraphs[x[0]].vs[x[1]]["label"], feature=subgraphs[x[0]].vs[x[1]]["feature"])

        new_edges_in_graph = list()
        new_edge_keys_in_graph = list()
        new_edge_labels_in_graph = list()
        new_edge_features_in_graph = list()
        
        # 用于跟踪已添加的边，避免重复
        added_edges = set()
        
        for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
            u = graph_vertex_mapping_reversed[(sg1, v1)]
            v = graph_vertex_mapping_reversed[(sg2, v2)]
            src_tgt = (u, v)
            
            # 获取对应的边特征
            edge_features = graph_edge_feature_mapping[(sg1, v1, sg2, v2)]
            
            # 为每种标签和特征组合创建一条边
            for i, (edge_label, edge_feature) in enumerate(zip(edge_labels, edge_features)):
                edge_key = (u, v, edge_label)
                if edge_key not in added_edges:
                    new_edges_in_graph.append(src_tgt)
                    new_edge_keys_in_graph.append(i)
                    new_edge_labels_in_graph.append(edge_label)
                    new_edge_features_in_graph.append(edge_feature)
                    added_edges.add(edge_key)
                    
        graph.add_edges(new_edges_in_graph)
        graph.es["label"] = new_edge_labels_in_graph
        graph.es["key"] = new_edge_keys_in_graph
        graph.es["feature"] = new_edge_features_in_graph

        return graph, graph_vertex_mapping, graph_vertex_mapping_reversed, new_edge_features_in_graph
    
    def _ensure_pattern_vertex_constraints(self, graph, graph_vertex_mapping, graph_vertex_mapping_reversed, subgraphs):
        """
        确保图中顶点满足模式图的标签和特征要求，如果不满足，则修改图中顶点的特征
        """
        # 收集模式图中所有需要的 (label, feature) 组合
        pattern_label_feature_pairs = list(zip(self.pattern.vs["label"], self.pattern.vs["feature"]))
        
        # 为图中每个顶点创建 (label, feature) 映射
        graph_vertex_label_feature_count = defaultdict(int)
        for v_label, v_feature in zip(graph.vs["label"], graph.vs["feature"]):
            graph_vertex_label_feature_count[(v_label, v_feature)] += 1
        
        # 统计每种标签的数量
        graph_vertex_label_count = defaultdict(int)
        for v_label in graph.vs["label"]:
            graph_vertex_label_count[v_label] += 1
            
        # 对每个模式顶点，确保图中有足够的匹配顶点
        for pattern_label, pattern_feature in pattern_label_feature_pairs:
            key = (pattern_label, pattern_feature)
            current_count = graph_vertex_label_feature_count[key]
            
            if current_count == 0:
                # 没有顶点匹配此 (label, feature) 组合，需要修改一个具有相同标签的顶点
                # 查找具有相同标签但不同特征的顶点
                for v_id, (v_label, v_feature) in enumerate(zip(graph.vs["label"], graph.vs["feature"])):
                    if v_label == pattern_label and v_feature != pattern_feature:
                        # 检查修改这个顶点是否会影响其他需求
                        old_key = (v_label, v_feature)
                        # 如果有多个顶点具有相同的 (label, feature)，可以安全地修改其中一个
                        if graph_vertex_label_feature_count[old_key] > 1:
                            graph.vs[v_id]["feature"] = pattern_feature
                            # 更新计数
                            graph_vertex_label_feature_count[old_key] -= 1
                            graph_vertex_label_feature_count[key] += 1
                            break
                        # 如果只有一个顶点具有这个 (label, feature) 但图中该标签的总数大于模式图中需要的数量，也可以修改
                        elif graph_vertex_label_count[v_label] > len([l for l, _ in pattern_label_feature_pairs if l == v_label]):
                            graph.vs[v_id]["feature"] = pattern_feature
                            # 更新计数
                            graph_vertex_label_feature_count[old_key] -= 1
                            graph_vertex_label_feature_count[key] += 1
                            break


    def generate(self, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
        alpha, max_pattern_counts=-1, max_subgraph=512, return_subisomorphisms=False):
        assert number_of_edges >= number_of_vertices - 1

        graph_pattern_valid = True
        if number_of_vertex_labels < self.number_of_pattern_vertex_labels:
            print("WARNING: the number of graph vertex labels (%d) is less than the number of pattern vertex labels (%d)." % (
                number_of_vertex_labels, self.number_of_pattern_vertex_labels))
            graph_pattern_valid = False
        if number_of_edge_labels < self.number_of_pattern_edge_labels:
            print("WARNING: the number of graph edge labels (%d) is less than the number of pattern edge labels (%d)." % (
                number_of_edge_labels, self.number_of_pattern_edge_labels))
            graph_pattern_valid = False
            
        if not graph_pattern_valid:
            # no subisomorphism in this setting
            # we can generate the graph randomly

            vertex_labels = generate_labels(number_of_vertices, number_of_vertex_labels)
            edge_labels = generate_labels(number_of_edges, number_of_edge_labels)
            vertex_features = [generate_vf_random_in_range(0, 527) for _ in range(number_of_vertices)]
            edge_features = [generate_ef_random_in_range(0, 319) for _ in range(number_of_edges)]
            graph = generate_tree(number_of_vertices, directed=True)
            graph_edge_label_mapping = defaultdict(set) # key: (0, v1, 0, v2), value: e_labels
            for e, edge in enumerate(graph.es):
                graph_edge_label_mapping[(0, edge.source, 0, edge.target)].add(edge_labels[e])
            ecount = graph.ecount()
            edge_keys = [0] * ecount

            # second, random add edges 
            new_edges = list()
            while ecount < number_of_edges:
                u = np.random.randint(0, number_of_vertices)
                v = np.random.randint(0, number_of_vertices)
                edge_label = edge_labels[ecount]
                # # we do not generate edges between two same vertices with same labels
                graph_edge_labels = graph_edge_label_mapping[(0, u, 0, v)]
                if edge_label in graph_edge_labels:
                    continue
                new_edges.append((u, v))
                edge_keys.append(len(graph_edge_labels))
                graph_edge_labels.add(edge_label)
                ecount += 1
            graph.add_edges(new_edges)
            graph.vs["label"] = vertex_labels
            graph.es["label"] = edge_labels
            graph.es["key"] = edge_keys
            graph.vs["feature"] = vertex_features
            graph.es["feature"] = edge_features
            
            metadata = {"counts": 0, "subisomorphisms": list()}
            return graph, metadata
        elif max_pattern_counts != -1 and number_of_edges * alpha > max_pattern_counts * self.number_of_pattern_edges:
            alpha = max_pattern_counts * self.number_of_pattern_edges / number_of_edges * DECAY
            return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                alpha=alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                return_subisomorphisms=return_subisomorphisms)
        else:
            # split the graph into small subgraphs to speed the subisomorphism searching
            subgraphs = list()
            number_of_subgraphs = math.ceil(number_of_vertices/max_subgraph)
            numbers_of_subgraph_vertices = np.array(np.random.dirichlet(
                [number_of_vertices/number_of_subgraphs] * number_of_subgraphs) * number_of_vertices, dtype=np.int)
            diff = number_of_vertices - numbers_of_subgraph_vertices.sum()
            numbers_of_subgraph_vertices[-1] += diff
            
            ecount = 0
            graph_vertex_label_mapping_reversed = defaultdict(list) # key: (sg, v_label), value: v_ids
            graph_edge_label_mapping = defaultdict(set) # key: (sg1, v1, sg2, v2), value: e_labels
            graph_vertex_feature_mapping = defaultdict(list)
            graph_edge_feature_mapping = defaultdict(set)

            
            for sg in range(number_of_subgraphs):
                # construct a directed tree
                number_of_subgraph_vertices = numbers_of_subgraph_vertices[sg]
                subgraph_vertex_labels = generate_labels(number_of_subgraph_vertices, number_of_vertex_labels)
                subgraph_edge_labels = generate_labels(number_of_subgraph_vertices-1, number_of_edge_labels) # tree label
                subgraph_vertex_features = [generate_vf_random_in_range(0, 527) for _ in range(number_of_subgraph_vertices)]
                subgraph_edge_features = [generate_ef_random_in_range(0, 319) for _ in range(number_of_subgraph_vertices-1)]
                subgraph = generate_tree(number_of_subgraph_vertices, directed=True)
                subgraph["sg"] = sg
                subgraph.vs["label"] = subgraph_vertex_labels
                subgraph.vs["feature"] = subgraph_vertex_features
                

                ecount += (number_of_subgraph_vertices-1)
                subgraphs.append(subgraph)
                for v_id, v_label in enumerate(subgraph_vertex_labels):
                    graph_vertex_label_mapping_reversed[(sg, v_label)].append(v_id)
                
                        
                for v_id_1, v_feature in enumerate(subgraph_vertex_features):
                    
                    graph_vertex_feature_mapping[(sg, v_feature)].append(v_id_1)
                for e, (v1, v2) in enumerate(subgraph.get_edgelist()):
                    graph_edge_label_mapping[(sg, v1, sg, v2)].add(subgraph_edge_labels[e])
                    graph_edge_feature_mapping[(sg, v1, sg, v2)].add(subgraph_edge_features[e])
                    # graph_edge_feature_mapping[(sg, e)].append(e_feature)
                subgraph.delete_edges(None)

                subgraph_pattern_valid = True
                subgraph_vertex_label_counter = Counter(subgraph_vertex_labels) # key; label, value: count
                for vertex_label, cnt in self.pattern_vertex_label_counter.items():
                    if subgraph_vertex_label_counter[vertex_label] < cnt:
                        subgraph_pattern_valid = False
                        break
                subgraph["pattern_valid"] = subgraph_pattern_valid

            for (sg1, sg2) in generate_tree(number_of_subgraphs, directed=True).get_edgelist():
                # add an edge between two subgraphs
                self.add_edges(subgraphs[sg1], subgraphs[sg2], graph_edge_label_mapping, graph_edge_feature_mapping, number_of_edge_labels, 1)
                ecount += 1
        
            invalid_cnt = 0
            while invalid_cnt < 10 and ecount < number_of_edges:
                sg1 = np.random.randint(0, number_of_subgraphs)
                sg2 = np.random.randint(0, number_of_subgraphs)
                diff = number_of_edges - ecount
                new_ecount=0
                if diff >= self.number_of_pattern_edges:
                    if subgraphs[sg1]["pattern_valid"] and np.random.rand() < alpha:
                        new_ecount = self.add_pattern(subgraphs[sg1], graph_vertex_label_mapping_reversed, graph_vertex_feature_mapping, graph_edge_label_mapping, graph_edge_feature_mapping)
                    else:
                        new_ecount = self.add_edges(subgraphs[sg1], subgraphs[sg2],
                            graph_edge_label_mapping, graph_edge_feature_mapping, number_of_edge_labels, self.number_of_pattern_edges)
                else:
                    new_ecount = self.add_edges(subgraphs[sg1], subgraphs[sg2],
                        graph_edge_label_mapping, graph_edge_feature_mapping, number_of_edge_labels, diff)
                if new_ecount == 0:
                    invalid_cnt += 1
                else:
                    invalid_cnt = 0
                    ecount += new_ecount
            if ecount < number_of_edges:
                alpha = alpha * ecount / number_of_edges * DECAY
                return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                    alpha=alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                    return_subisomorphisms=return_subisomorphisms)
            
            # self.update_subgraphs(subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping)
            # graph, graph_vertex_mapping, graph_vertex_mapping_reversed, new_edge_features_in_graph = self.merge_subgraphs(subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping)
            
            # # 确保图中顶点满足模式图的标签和特征要求
            # self._ensure_pattern_vertex_constraints(graph, graph_vertex_mapping, graph_vertex_mapping_reversed, subgraphs)
            
            # if return_subisomorphisms:
            #     subisomorphisms = list()
            #     for sg, subgraph in enumerate(subgraphs):
            #         for subisomorphism in self.pattern_checker.get_subisomorphisms(subgraph, self.pattern):
            #             subisomorphism = [graph_vertex_mapping_reversed[(sg, v)] for v in subisomorphism]
            #             subisomorphisms.append(subisomorphism)
            #     metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}
            # else:
            #     counts = 0
            #     for subgraph in subgraphs:
            #         counts += self.pattern_checker.count_subisomorphisms(subgraph, self.pattern)
            #     metadata = {"counts": counts, "subisomorphisms": list()}
            self.update_subgraphs(subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping)
            graph, graph_vertex_mapping, graph_vertex_mapping_reversed, new_edge_features_in_graph = self.merge_subgraphs(subgraphs, graph_edge_label_mapping, graph_edge_feature_mapping)
            
            # 确保图中顶点满足模式图的标签和特征要求
            self._ensure_pattern_vertex_constraints(graph, graph_vertex_mapping, graph_vertex_mapping_reversed, subgraphs)
            
            if return_subisomorphisms:
                subisomorphisms = self.pattern_checker.get_subisomorphisms(graph, self.pattern)
                metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}
            else:
                counts = self.pattern_checker.count_subisomorphisms(graph, self.pattern)
                metadata = {"counts": counts, "subisomorphisms": list()}
            if metadata["counts"] > max_pattern_counts:
                alpha = alpha * max_pattern_counts / metadata["counts"] * DECAY
                return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                    alpha=alpha, max_subgraph=max_subgraph, max_pattern_counts=max_pattern_counts,
                    return_subisomorphisms=return_subisomorphisms)
            
            # assert(metadata["counts"] == self.pattern_checker.count_subisomorphisms(graph, self.pattern))
            
            return graph, metadata

    

    def _ensure_pattern_vertex_constraints(self, graph, graph_vertex_mapping, graph_vertex_mapping_reversed, subgraphs):
        """
        确保图中顶点满足模式图的标签和特征要求，如果不满足，则修改图中顶点的特征
        """
        # 收集模式图中所有需要的 (label, feature) 组合
        pattern_label_feature_pairs = list(zip(self.pattern.vs["label"], self.pattern.vs["feature"]))
        
        # 创建一个字典来存储 (label, feature) 组合到顶点 ID 的映射
        graph_vertex_label_feature_mapping = defaultdict(list)
        for v_id, (v_label, v_feature) in enumerate(zip(graph.vs["label"], graph.vs["feature"])):
            graph_vertex_label_feature_mapping[(v_label, v_feature)].append(v_id)
        
        # 对于模式图中的每个顶点，确保图中有对应标签和特征的顶点
        for pattern_label, pattern_feature in pattern_label_feature_pairs:
            key = (pattern_label, pattern_feature)
            # 如果找不到同时匹配标签和特征的顶点
            if key not in graph_vertex_label_feature_mapping or not graph_vertex_label_feature_mapping[key]:
                # 查找一个具有相同标签但特征不同的顶点，并修改其特征
                for v_id, (v_label, v_feature) in enumerate(zip(graph.vs["label"], graph.vs["feature"])):
                    if v_label == pattern_label and v_feature != pattern_feature:
                        # 检查修改这个顶点是否会影响其他模式顶点的需求
                        old_key = (v_label, v_feature)
                        # 如果有多个顶点具有相同的 (label, feature)，可以安全地修改其中一个
                        if len(graph_vertex_label_feature_mapping[old_key]) > 1:
                            graph.vs[v_id]["feature"] = pattern_feature
                            # 更新映射
                            graph_vertex_label_feature_mapping[old_key].remove(v_id)
                            graph_vertex_label_feature_mapping[key].append(v_id)
                            break
                        # 如果只有一个顶点具有这个 (label, feature) 但图中该标签的总数大于模式图中需要的数量，也可以修改
                        elif len([l for l, _ in pattern_label_feature_pairs if l == v_label]) < len([l for l in graph.vs["label"] if l == v_label]):
                            graph.vs[v_id]["feature"] = pattern_feature
                            # 更新映射
                            graph_vertex_label_feature_mapping[old_key].remove(v_id)
                            graph_vertex_label_feature_mapping[key].append(v_id)
                            break
    
    
    def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_vertex_feature_mapping,\
                    graph_edge_label_mapping, graph_edge_feature_mapping):
        sg = subgraph["sg"]
        subisomorphism = list()
        
        # 创建一个字典来存储 (label, feature) 组合到顶点 ID 的映射
        graph_vertex_label_feature_mapping_reversed = defaultdict(list)
        for v_id, (v_label, v_feature) in enumerate(zip(subgraph.vs["label"], subgraph.vs["feature"])):
            graph_vertex_label_feature_mapping_reversed[(sg, v_label, v_feature)].append(v_id)
        
        # 根据模式图的顶点标签和特征，在子图中选择对应标签和特征的顶点
        for vertex_label, vertex_feature in zip(self.pattern.vs["label"], self.pattern.vs["feature"]):
            key = (sg, vertex_label, vertex_feature)
            if key in graph_vertex_label_feature_mapping_reversed and graph_vertex_label_feature_mapping_reversed[key]:
                # 找到同时匹配标签和特征的顶点
                subisomorphism.append(sample_element(graph_vertex_label_feature_mapping_reversed[key]))
            else:
                # 如果找不到同时匹配标签和特征的顶点，则选择一个具有相同标签的顶点
                # 实际的特征修改将在合并所有子图后进行
                subisomorphism.append(sample_element(graph_vertex_label_mapping_reversed[(sg, vertex_label)]))
        
        new_ecount = 0
        for (pattern_u, pattern_v), pattern_edge_labels in self.pattern_edge_label_mapping.items():
            graph_u = subisomorphism[pattern_u]
            graph_v = subisomorphism[pattern_v]
            graph_edge_labels = graph_edge_label_mapping[(sg, graph_u, sg, graph_v)]
            edge_label_diff = pattern_edge_labels - graph_edge_labels
            
            for edge_label in edge_label_diff:
                graph_edge_labels.add(edge_label)
            new_ecount += len(edge_label_diff)
        for (pattern_u, pattern_v), pattern_edge_features in self.pattern_edge_feature_mapping.items():
            graph_u = subisomorphism[pattern_u]
            graph_v = subisomorphism[pattern_v]
            graph_edge_features = graph_edge_feature_mapping[(sg, graph_u, sg, graph_v)]
            edge_feature_diff = pattern_edge_features - graph_edge_features
            
            for edge_feature in edge_feature_diff:
                graph_edge_features.add(edge_feature)
        return new_ecount
    
    def add_edges(self, subgraph1, subgraph2, graph_edge_label_mapping, graph_edge_feature_mapping,
                   graph_number_of_edge_labels, number_of_edges):
        # 添加模板特征映射信息
        sg1 = subgraph1["sg"]
        sg2 = subgraph2["sg"]
        g1_vcount = subgraph1.vcount()
        g2_vcount = subgraph2.vcount()
        new_ecount = 0
        invalid_cnt = 0
        
        while invalid_cnt < 10 and new_ecount < number_of_edges:
            v1 = np.random.randint(0, g1_vcount)
            v2 = np.random.randint(0, g2_vcount)
            edge_label = np.random.randint(0, graph_number_of_edge_labels)
            edge_feature = generate_ef_random_in_range(0, 319)
            
            if get_direction():
                x = (subgraph1.vs[v1]["label"], subgraph2.vs[v2]["label"])
                y = (sg1, v1, sg2, v2)
            else:
                x = (subgraph2.vs[v2]["label"], subgraph1.vs[v1]["label"])
                y = (sg2, v2, sg1, v1)
                
            if edge_label in self.pattern_nec_tree_vertex_edge_label_mapping[x]:
                invalid_cnt += 1
                continue
                
            graph_edge_labels = graph_edge_label_mapping[y]
            graph_edge_features = graph_edge_feature_mapping[y]
            if edge_label in graph_edge_labels:
                invalid_cnt += 1
                continue
            graph_edge_labels.add(edge_label)
            graph_edge_features.add(edge_feature)
            invalid_cnt = 0
            new_ecount += 1
        return new_ecount


def generate_graphs(pattern, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels, \
    alpha, max_pattern_counts, max_subgraph, return_subisomorphisms, number_of_graphs):
    graph_generator = GraphGenerator(pattern)
    results = list()
    for g in range(number_of_graphs):
        graph, metadata = graph_generator.generate(
                number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                return_subisomorphisms=return_subisomorphisms)
        print("%d/%d" % (g+1, number_of_graphs), "number of subisomorphisms: %d" % (metadata["counts"]))
        results.append((graph, metadata))
    return results

def draw(graph, pattern, subisomorphisms):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    ig.plot(graph, "graph.png")
    ig.plot(pattern, "pattern.png")

    graph_pattern = graph.copy()
    pal = ig.drawing.colors.ClusterColoringPalette(len(subisomorphisms)+1)
    # graph_pattern.vs["color"] = pal.get(0)
    # graph_pattern.es["color"] = pal.get(0)
    for i, subisomorphism in enumerate(subisomorphisms):
        for pattern_vertex, graph_vertex in enumerate(subisomorphism):
            graph_pattern.vs[graph_vertex]["color"] = pal.get(i+1)
            graph_edges = graph.incident(graph_vertex)
            graph_edge_dict = dict()
            for graph_edge in graph_edges:
                graph_edge = graph_pattern.es[graph_edge]
                graph_edge_dict[(graph_edge.target, graph_edge["label"])] = graph_edge
            for pattern_edge in pattern.incident(pattern_vertex):
                pattern_edge = pattern.es[pattern_edge]
                pattern_tgt = pattern_edge.target
                edge_label = pattern_edge["label"]
                graph_edge_dict[(subisomorphism[pattern_tgt], edge_label)]["color"] = pal.get(i+1)
    ig.plot(graph_pattern, "graph_pattern.png", palette=pal)

    plt.subplot(1, 3, 1)
    plt.imshow(plt.imread("graph.png"))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(plt.imread("pattern.png"))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(plt.imread("graph_pattern.png"))
    plt.axis("off")

    plt.text(0, 0, "#isomorphic subgraphs: %d" % (len(subisomorphisms)))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_of_vertices", type=int, default=2048)
    parser.add_argument("--number_of_edges", type=int, default=2048*4)
    parser.add_argument("--number_of_vertex_labels", type=int, default=128)
    parser.add_argument("--number_of_edge_labels", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_pattern_counts", type=float, default=2048)
    parser.add_argument("--return_subisomorphisms", type=str2bool, default=False)
    parser.add_argument("--max_subgraph", type=int, default=512)
    parser.add_argument("--number_of_graphs", type=int, default=10)
    parser.add_argument("--pattern_path", type=str,default=r"patterns/P_N3_E3_NL2_EL2_0.gml")
    parser.add_argument("--save_graph_dir", type=str, default="graphs")
    parser.add_argument("--save_metadata_dir", type=str, default="metadata")
    parser.add_argument("--save_png", type=str2bool, default=False)
    parser.add_argument("--show_img", type=str2bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    try:
        pattern = ig.read(args.pattern_path)
        pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
        pattern.es["label"] = [int(x) for x in pattern.es["label"]]
        pattern.es["key"] = [int(x) for x in pattern.es["key"]]
    except BaseException as e:
        print(e)
        pattern = ig.Graph(directed=True)
        pattern.vs["label"] = []
        pattern.es["label"] = []
        pattern.es["key"] = []

    results = generate_graphs(pattern,
        args.number_of_vertices, args.number_of_edges,
        args.number_of_vertex_labels, args.number_of_edge_labels,
        args.alpha, args.max_pattern_counts, args.max_subgraph,
        args.return_subisomorphisms, args.number_of_graphs)

    if args.save_graph_dir:
        os.makedirs(args.save_graph_dir, exist_ok=True)
        save_graph_dir = os.path.join(args.save_graph_dir, os.path.splitext(os.path.basename(args.pattern_path))[0])
        os.makedirs(save_graph_dir, exist_ok=True)
        if args.save_metadata_dir:
            os.makedirs(args.save_metadata_dir, exist_ok=True)
            save_metadata_dir = os.path.join(args.save_metadata_dir, os.path.splitext(os.path.basename(args.pattern_path))[0])
            os.makedirs(save_metadata_dir, exist_ok=True)
        for g, (graph, metadata) in enumerate(results):
            graph_id = "G_N%d_E%d_NL%d_EL%d_%d" % (
                graph.vcount(), graph.ecount(), args.number_of_vertex_labels, args.number_of_edge_labels, g)
            graph_filename = os.path.join(save_graph_dir, graph_id)
            graph.write(graph_filename + ".gml")
            if args.save_metadata_dir:
                metadata_filename = os.path.join(save_metadata_dir, graph_id)
                with open(metadata_filename + ".meta", "w") as f:
                    json.dump(metadata, f)
            if args.save_png:
                ig.plot(graph, graph_filename + ".png")
            if args.show_img:
                draw(graph, pattern, metadata["subisomorphisms"])
