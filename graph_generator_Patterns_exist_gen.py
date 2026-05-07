import igraph as ig
import numpy as np
import argparse
import os
import math
import json
from collections import Counter, defaultdict
from utils_Patterns_exist_gen import generate_edge_label_by_device_types,generate_labels, generate_tree, get_direction, powerset, sample_element, str2bool, retrieve_multiple_edges,generate_vf_random_in_range_list,generate_ef_random_in_range_list,generate_ef_random_in_range
from pattern_checker import PatternChecker
from pattern_generator_P import generate_patterns
from time import time
import copy

DECAY = 0.954 # 2 sigma

class NEC(object):
    def __init__(self, data=None, adj=None, inter_adj=None, vertex_label=None, is_clique=False, nec_id=0):
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
        self.NEC_by_vertex_index = [None] * vcount
    
    def add_nec(self, nec):
        adj = nec.adj
        vertex_label = nec.vertex_label
        is_clique = nec.is_clique

        nec.nec_id = self.tree.vcount()
        self.tree.add_vertex(label=vertex_label)

        if adj not in self.NEC_by_adj:
            self.NEC_by_adj[adj] = list()
        self.NEC_by_adj[adj].append(nec)

        if vertex_label not in self.NEC_by_vertex_label:
            self.NEC_by_vertex_label[vertex_label] = list()
        self.NEC_by_vertex_label[vertex_label].append(nec)
        
        for vertex_index in nec:
            self.NEC_by_vertex_index[vertex_index] = nec
    
    def add_edge(self, source, target, edge_label):
        self.tree.add_edge(source, target, label=edge_label)
class GraphGenerator(object):
    def __init__(self, pattern):
        self.pattern = pattern if pattern else ig.Graph(directed=True)
        self.number_of_pattern_vertices = self.pattern.vcount()
        self.number_of_pattern_edges = self.pattern.ecount()
        
        # 确保pattern的边标签是字符型
        self.pattern.vs["label"] = [str(x) for x in self.pattern.vs["label"]]
        self.pattern.es["label"] = [str(x) for x in self.pattern.es["label"]]
        
        self.pattern_vertex_label_counter = Counter(self.pattern.vs["label"])
        self.pattern_edge_label_counter = Counter(self.pattern.es["label"])
        self.pattern_vertex_label_list = self.pattern.vs["label"]
        self.pattern_edge_label_list = self.pattern.es["label"]

        self.pattern_edge_label_mapping = defaultdict(set)
        self.pattern_vertex_edge_label_mapping = defaultdict(set)
        for edge in self.pattern.es:
            self.pattern_edge_label_mapping[edge.tuple].add(edge["label"])
            key = (self.pattern.vs[edge.source]["label"], self.pattern.vs[edge.target]["label"])
            self.pattern_vertex_edge_label_mapping[key].add(edge["label"])

        # 使用实际使用的标签种类数
        self.number_of_pattern_vertex_labels = len(set(self.pattern.vs["label"]))
        self.number_of_pattern_edge_labels = len(set(self.pattern.es["label"]))

        self.pattern_nec_tree = self.rewrite_to_nec_tree()
        self.pattern_nec_tree_vertex_edge_label_mapping = defaultdict(set)
        for edge in self.pattern_nec_tree.tree.es:
            key = (self.pattern_nec_tree.tree.vs[edge.source]["label"], self.pattern_nec_tree.tree.vs[edge.target]["label"])
            self.pattern_nec_tree_vertex_edge_label_mapping[key].add(edge["label"])

        self.pattern_checker = PatternChecker()

    def choose_start_q_vertex(self):
        vs = list()
        for v in self.pattern.vs:
            freq = self.pattern_vertex_label_counter[v["label"]]
            deg = v.degree()
            vs.append((freq/deg, v.index))
        vs.sort()
        return vs[0][1]
    
    def find_cliques(self, edges):
        cliques = list()
        vs = set()
        for edge in edges:
            vs.update(edge)
        if len(vs) < 2:
            return cliques
        for clique_vs in powerset(vs, min_size=2):
            n = len(clique_vs)
            in_degrees = Counter()
            out_degrees = Counter()
            for edge in edges:
                if edge[0] in clique_vs and edge[1] in clique_vs:
                    out_degrees[edge[0]] += 1
                    in_degrees[edge[1]] += 1
            in_d = in_degrees[clique_vs[0]]
            out_d = out_degrees[clique_vs[0]]
            if in_d == 0 or (in_d % (n-1) != 0):
                continue
            if in_d != out_d:
                continue
            if not (all([in_d == in_degree for in_degree in in_degrees.values()]) \
                    and all([out_d == out_degree for out_degree in out_degrees.values()])):
                continue
            cliques.append(clique_vs)
        return cliques

    def find_necs(self, group):
        groups_by_adj = defaultdict(list)
        for v in group:
            adj = list()
            for out_e in self.pattern.incident(v, mode=ig.OUT):
                edge = self.pattern.es[out_e]
                u = edge.target
                adj.append((ig.OUT, edge["label"], u))
            for in_e in self.pattern.incident(v, mode=ig.IN):
                edge = self.pattern.es[in_e]
                u = edge.source
                adj.append((ig.IN, edge["label"], u))
            adj = tuple(sorted(adj))
            groups_by_adj[adj].append(v)
        
        singleton_group_mapping = dict()
        necs = list()
        for adj, vs in groups_by_adj.items():
            if len(vs) > 1:
                necs.append(NEC(sorted(vs), adj=adj, is_clique=False))
            else:
                singleton_group_mapping[vs[0]] = adj
        
        groups_by_degree = defaultdict(list)
        for v, adj in singleton_group_mapping.items():
            in_degree = len([x[0] == ig.IN for x in adj])
            out_degree = len(adj) - in_degree
            groups_by_degree[(in_degree, out_degree)].append(v)
        for key, vs in groups_by_degree.items():
            if len(vs) == 1:
                necs.append(NEC(vs, adj=singleton_group_mapping[vs[0]], is_clique=False))
            else:
                inter_edges_by_edge_labels = defaultdict(set)
                edge_label_set = set()
                for v in vs:
                    for x in singleton_group_mapping[v]:
                        if x[2] in vs:
                            if x[0] == ig.OUT:
                                src_tgt = (v, x[2])
                            else:
                                src_tgt = (x[2], v)
                            inter_edges_by_edge_labels[x[1]].add(src_tgt)
                
                cliques_by_edge_labels = dict()
                for edge_label, edges in inter_edges_by_edge_labels.items():
                    cliques = self.find_cliques(edges)
                    if len(cliques) > 0:
                        cliques_by_edge_labels[edge_label] = set(cliques)
                if len(cliques_by_edge_labels) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                        
                mixed_cliques = set.intersection(*cliques_by_edge_labels.values())
                if len(mixed_cliques) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                
                valid_cliques = dict()
                for clique in mixed_cliques:
                    inter_adjs = dict()
                    outer_adjs = dict()
                    for v in clique:
                        adj = singleton_group_mapping[v]
                        inter_adjs[v] = sorted([(x[0], x[1]) for x in adj if x[2] in clique])
                        outer_adjs[v] = sorted([x for x in adj if x[2] not in clique])
                    
                    o_adj = next(iter(outer_adjs.values()))
                    if not all([o_adj == outer_adj for outer_adj in outer_adjs.values()]):
                        continue    
                    i_adj = next(iter(inter_adjs.values()))
                    valid_cliques[clique] = (o_adj, i_adj)
                if len(valid_cliques) == 0:
                    for v in vs:
                        necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
                    continue
                
                final_cliques = list()
                for valid_clique in sorted(valid_cliques.keys(), key=lambda x: (-len(x), x)):
                    is_subclique = False
                    for final_clique in final_cliques:
                        if final_clique.issuperset(valid_clique):
                            is_subclique = True
                            break
                    if not is_subclique:
                        final_cliques.append(set(valid_clique))
                
                for final_clique in final_cliques:
                    final_clique = sorted(final_clique)
                    outer_adj, inter_adj = valid_cliques[tuple(final_clique)]
                    necs.append(NEC(final_clique, adj=tuple(outer_adj), inter_adj=tuple(inter_adj), is_clique=True))
                
                for v in set(vs).difference(set.union(*final_cliques)):
                    necs.append(NEC([v], adj=singleton_group_mapping[v], is_clique=False))
        return necs
    def rewrite_to_nec_tree(self):
        nec_tree = NECTree(self.number_of_pattern_vertices, directed=True)

        start_v = self.choose_start_q_vertex()
        visited = [0] * self.number_of_pattern_vertices
        visited[start_v] = 1
        adj = list()
        out_edges = retrieve_multiple_edges(self.pattern, source=start_v)
        in_edges = retrieve_multiple_edges(self.pattern, target=start_v)
        adj.extend([(ig.OUT, edge["label"], edge.target) for edge in out_edges])
        adj.extend([(ig.IN, edge["label"], edge.source) for edge in in_edges])
        adj = tuple(sorted(adj))
        
        root = NEC(data=[start_v], adj=adj, vertex_label=self.pattern.vs[start_v]["label"], is_clique=False, nec_id=0)
        nec_tree.add_nec(root)

        v_current = list()
        v_next = [root]

        while len(v_next) > 0:
            v_current, v_next = v_next, list()
            for nec in v_current:
                groups = defaultdict(list)
                for v in nec:
                    out_edges = retrieve_multiple_edges(self.pattern, source=v)
                    in_edges = retrieve_multiple_edges(self.pattern, target=v)
                    for edge in sorted(out_edges, key=lambda x: x["label"]):
                        u = edge.target
                        if not visited[u]:
                            key = (ig.OUT, edge["label"], self.pattern.vs[u]["label"])
                            groups[key].append(u)
                            visited[u] = 1
                    for edge in sorted(in_edges, key=lambda x: x["label"]):
                        u = edge.source
                        if not visited[u]:
                            key = (ig.IN, edge["label"], self.pattern.vs[u]["label"])
                            groups[key].append(u)
                            visited[u] = 1
                for key, group in groups.items():
                    mode, edge_label, vertex_label = key
                    new_necs = self.find_necs(group)
                    for new_nec in new_necs:
                        new_nec.vertex_label = vertex_label
                        nec_tree.add_nec(new_nec)
                        if mode == ig.OUT:
                            nec_tree.add_edge(nec.nec_id, new_nec.nec_id, edge_label=edge_label)
                        else:
                            nec_tree.add_edge(new_nec.nec_id, nec.nec_id, edge_label=edge_label)
                        v_next.append(new_nec)
            v_next.sort(key=lambda x: x.vertex_label)
        return nec_tree

    # def update_subgraphs(self, subgraphs, graph_edge_label_mapping):
    #     new_edges_in_subgraphs = [list() for i in range(len(subgraphs))]
    #     new_edge_labels_in_subgraphs = [list() for i in range(len(subgraphs))]
    #     subgraphs_vlabels = [subgraph.vs["label"] for subgraph in subgraphs]
        
    #     for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
    #         # 避免自环边
    #         if sg1 == sg2 and v1 == v2:
    #             continue
                
    #         if sg1 == sg2:
    #             src_tgt = (v1, v2)
    #             # 确保不是自环边
    #             if src_tgt[0] != src_tgt[1]:
    #                 key = (subgraphs_vlabels[sg1][v1], subgraphs_vlabels[sg2][v2])
    #                 # 确保节点标签是整数类型
    #                 source_label = subgraphs_vlabels[sg1][v1]
    #                 target_label = subgraphs_vlabels[sg2][v2]
    #                 if isinstance(source_label, str):
    #                     source_label = int(source_label)
    #                 if isinstance(target_label, str):
    #                     target_label = int(target_label)
    #                 key = (source_label, target_label)
    #                 pattern_edge_labels = self.pattern_vertex_edge_label_mapping[key]
    #                 edge_labels = [edge_label for edge_label in edge_labels if edge_label in pattern_edge_labels] 
    #                 new_edges_in_subgraphs[sg1].extend([src_tgt] * len(edge_labels))
    #                 new_edge_labels_in_subgraphs[sg1].extend(edge_labels)
        
    #     for sg, subgraph in enumerate(subgraphs):
    #         # 过滤掉自环边
    #         valid_edges = []
    #         valid_labels = []
    #         for edge, label in zip(new_edges_in_subgraphs[sg], new_edge_labels_in_subgraphs[sg]):
    #             if edge[0] != edge[1]:  # 不是自环边
    #                 valid_edges.append(edge)
    #                 valid_labels.append(label)
            
    #         if valid_edges:
    #             subgraph.add_edges(valid_edges)
    #             # 检查是否有边标签需要设置，避免空序列错误
    #             if valid_labels:
    #                 subgraph.es["label"] = valid_labels
    #             else:
    #                 # 如果没有边标签，确保现有的边也有标签（如果有的话）
    #                 if subgraph.ecount() > 0 and "label" not in subgraph.edge_attributes():
    #                     # 设置默认标签
    #                     subgraph.es["label"] = ["0"] * subgraph.ecount()
    def update_subgraphs(self, subgraphs, graph_edge_label_mapping):
        new_edges_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_labels_in_subgraphs = [list() for i in range(len(subgraphs))]
        subgraphs_vlabels = [subgraph.vs["label"] for subgraph in subgraphs]
        
        for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
            # 避免自环边
            if sg1 == sg2 and v1 == v2:
                continue
                
            if sg1 == sg2:
                src_tgt = (v1, v2)
                # 确保不是自环边
                if src_tgt[0] != src_tgt[1]:
                    # 处理边标签
                    for edge_label in edge_labels:
                        # 检查是否已经添加了这条边
                        edge_exists = False
                        for i, (existing_edge, existing_label) in enumerate(zip(new_edges_in_subgraphs[sg1], new_edge_labels_in_subgraphs[sg1])):
                            if existing_edge == src_tgt and existing_label == edge_label:
                                edge_exists = True
                                break
                        
                        if not edge_exists:
                            new_edges_in_subgraphs[sg1].append(src_tgt)
                            new_edge_labels_in_subgraphs[sg1].append(edge_label)
        
        for sg, subgraph in enumerate(subgraphs):
            # 过滤掉自环边
            valid_edges = []
            valid_labels = []
            for edge, label in zip(new_edges_in_subgraphs[sg], new_edge_labels_in_subgraphs[sg]):
                if edge[0] != edge[1]:  # 不是自环边
                    valid_edges.append(edge)
                    valid_labels.append(label)
            
            if valid_edges:
                subgraph.add_edges(valid_edges)
                # 检查是否有边标签需要设置，避免空序列错误
                if valid_labels:
                    subgraph.es["label"] = valid_labels
                else:
                    # 如果没有边标签，确保现有的边也有标签（如果有的话）
                    if subgraph.ecount() > 0 and "label" not in subgraph.edge_attributes():
                        # 设置默认标签
                        subgraph.es["label"] = ["00100199"] * subgraph.ecount()

    def merge_subgraphs(self, subgraphs, graph_edge_label_mapping):
        graph = ig.Graph(directed=True)
        graph_vertex_mapping = list()
        graph_vertex_mapping_reversed = dict()
        for sg, subgraph in enumerate(subgraphs):
            for v_id in range(subgraph.vcount()):
                graph_vertex_mapping.append((sg, v_id))
        np.random.shuffle(graph_vertex_mapping)

        for v_id, x in enumerate(graph_vertex_mapping):
            graph_vertex_mapping_reversed[x] = v_id
            graph.add_vertex(label=subgraphs[x[0]].vs[x[1]]["label"])

        new_edges_in_graph = list()
        new_edge_keys_in_graph = list()
        new_edge_labels_in_graph = list()
        for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
            u = graph_vertex_mapping_reversed[(sg1, v1)]
            v = graph_vertex_mapping_reversed[(sg2, v2)]
            src_tgt = (u,v)
            new_edges_in_graph.extend([src_tgt] * len(edge_labels))
            new_edge_keys_in_graph.extend(range(len(edge_labels)))
            new_edge_labels_in_graph.extend(edge_labels)
        graph.add_edges(new_edges_in_graph)
        graph.es["label"] = new_edge_labels_in_graph
        graph.es["key"] = new_edge_keys_in_graph
        
        return graph, graph_vertex_mapping, graph_vertex_mapping_reversed
    

    def _generate_fallback(self, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels):
        """备用生成方案"""
        # 生成随机图，确保边标签为字符型
        vertex_labels = [str(x) for x in generate_vf_random_in_range_list(number_of_vertices)]
        edge_labels = [str(x) for x in generate_ef_random_in_range_list(number_of_edges)]  # 转换为字符型
        
        graph = generate_tree(number_of_vertices, directed=True)
        graph_edge_label_mapping = defaultdict(set)
        for e, edge in enumerate(graph.es):
            graph_edge_label_mapping[(0, edge.source, 0, edge.target)].add(edge_labels[e])
        ecount = graph.ecount()
        edge_keys = [0] * ecount

        new_edges = list()
        while ecount < number_of_edges:
            u = np.random.randint(0, number_of_vertices)
            v = np.random.randint(0, number_of_vertices)
            edge_label = edge_labels[ecount]
            graph_edge_labels = graph_edge_label_mapping[(0, u, 0, v)]
            if edge_label in graph_edge_labels:
                continue
            new_edges.append((u, v))
            edge_keys.append(len(graph_edge_labels))
            graph_edge_labels.add(edge_label)
            ecount += 1
        graph.add_edges(new_edges)
        graph.vs["label"] = vertex_labels
        graph.es["label"] = edge_labels  # 字符型边标签
        graph.es["key"] = edge_keys
        subisomorphisms = list()
        metadata = {"counts": 0, "subisomorphisms": subisomorphisms}
        return graph, metadata

    def generate(self, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
            alpha, max_pattern_counts=-1, max_subgraph=512, return_subisomorphisms=False, max_recursion=10):
        assert number_of_edges >= number_of_vertices - 1

        if not hasattr(self, '_recursion_depth'):
            self._recursion_depth = 0
        else:
            self._recursion_depth += 1
        
        if self._recursion_depth > max_recursion:
            return self._generate_fallback(number_of_vertices, number_of_edges, 
                                        number_of_vertex_labels, number_of_edge_labels)

        graph_pattern_valid = True
        if number_of_vertex_labels < self.number_of_pattern_vertex_labels:
            print("WARNING: the number of graph vertex labels (%d) is less than the number of pattern vertex labels (%d)." % (
                number_of_vertex_labels, self.number_of_pattern_vertex_labels))
        if number_of_edge_labels < self.number_of_pattern_edge_labels:
            print("WARNING: the number of graph edge labels (%d) is less than the number of pattern edge labels (%d)." % (
                number_of_edge_labels, self.number_of_pattern_edge_labels))
            
        if not graph_pattern_valid:
            return self._generate_fallback(number_of_vertices, number_of_edges, 
                                        number_of_vertex_labels, number_of_edge_labels)
        elif max_pattern_counts != -1 and number_of_edges * alpha > max_pattern_counts * self.number_of_pattern_edges:
            alpha = max_pattern_counts * self.number_of_pattern_edges / number_of_edges * DECAY
            return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                alpha=alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                return_subisomorphisms=return_subisomorphisms)
        else:
            subgraphs = list()
            number_of_subgraphs = math.ceil(number_of_vertices/max_subgraph)
            numbers_of_subgraph_vertices = np.array(np.random.dirichlet(
                [number_of_vertices/number_of_subgraphs] * number_of_subgraphs) * number_of_vertices, dtype=np.int)
            diff = number_of_vertices - numbers_of_subgraph_vertices.sum()
            numbers_of_subgraph_vertices[-1] += diff
            
            ecount = 0
            graph_vertex_label_mapping_reversed = defaultdict(list)
            graph_edge_label_mapping = defaultdict(set)
            for sg in range(number_of_subgraphs):
                number_of_subgraph_vertices = numbers_of_subgraph_vertices[sg]
                
                # 确保顶点标签和边标签为字符型
                subgraph_vertex_labels = [str(x) for x in generate_vf_random_in_range_list(number_of_subgraph_vertices)]
                subgraph_edge_labels = [str(x) for x in generate_ef_random_in_range_list(number_of_subgraph_vertices-1)]  # 字符型
                
                # 从pattern中随机选择标签
                num_random_elements = np.random.randint(0, len(subgraph_vertex_labels) + 1)
                random_indices = np.random.choice(len(subgraph_vertex_labels), size=num_random_elements, replace=False)
                for idx in random_indices:
                    subgraph_vertex_labels[idx] = self.pattern_vertex_label_list[int(np.random.randint(0, len(self.pattern_vertex_label_list)))]
                
                num_random_elements = np.random.randint(0, len(subgraph_edge_labels) + 1)
                random_indices = np.random.choice(len(subgraph_edge_labels), size=num_random_elements, replace=False)
                for idx in random_indices:
                    subgraph_edge_labels[idx] = self.pattern_edge_label_list[int(np.random.randint(0, len(self.pattern_edge_label_list)))]
                
                subgraph = generate_tree(number_of_subgraph_vertices, directed=True)
                subgraph["sg"] = sg
                subgraph.vs["label"] = subgraph_vertex_labels

                    # 确保边标签属性存在
                if "label" not in subgraph.es.attributes():
                    if subgraph.ecount() > 0:
                        subgraph.es["label"] = ["0"] * subgraph.ecount()

                
                ecount += (number_of_subgraph_vertices-1)
                subgraphs.append(subgraph)
                    # 确保每个子图至少有一条边
                for sg, subgraph in enumerate(subgraphs):
                    if subgraph.ecount() == 0 and numbers_of_subgraph_vertices[sg] > 1:
                        # 为没有边的子图添加至少一条边
                        u = np.random.randint(0, numbers_of_subgraph_vertices[sg])
                        v = np.random.randint(0, numbers_of_subgraph_vertices[sg])
                        # 确保源和目标不同，避免自环边
                        while u == v:
                            v = np.random.randint(0, numbers_of_subgraph_vertices[sg])
                        
                        # 生成边标签
                        source_label = subgraph.vs[u]["label"]
                        target_label = subgraph.vs[v]["label"]
                        # 确保source_label和target_label是整数类型
                        if isinstance(source_label, str):
                            source_label = int(source_label)
                        if isinstance(target_label, str):
                            target_label = int(target_label)
                        edge_label = str(generate_edge_label_by_device_types(source_label, target_label))
                        
                        # 添加到图边标签映射
                        graph_edge_label_mapping[(sg, u, sg, v)].add(edge_label)
                        # 同时添加边到子图
                        subgraph.add_edges([(u, v)])
                        subgraph.es[subgraph.ecount()-1]["label"] = edge_label
                for v_id, v_label in enumerate(subgraph_vertex_labels):
                    graph_vertex_label_mapping_reversed[(sg, v_label)].append(v_id)
                for e, (v1, v2) in enumerate(subgraph.get_edgelist()):
                    graph_edge_label_mapping[(sg, v1, sg, v2)].add(subgraph_edge_labels[e])
                subgraph.delete_edges(None)

                subgraph_pattern_valid = True
                subgraph_vertex_label_counter = Counter(subgraph_vertex_labels)
                for vertex_label, cnt in self.pattern_vertex_label_counter.items():
                    if subgraph_vertex_label_counter[vertex_label] < cnt:
                        subgraph_pattern_valid = False
                        break
                subgraph["pattern_valid"] = subgraph_pattern_valid

            for (sg1, sg2) in generate_tree(number_of_subgraphs, directed=True).get_edgelist():
                self.add_edges(subgraphs[sg1], subgraphs[sg2], graph_edge_label_mapping, number_of_edge_labels, 1)
                ecount += 1
            invalid_cnt = 0
            while invalid_cnt < 10 and ecount < number_of_edges:
                sg1 = np.random.randint(0, number_of_subgraphs)
                sg2 = np.random.randint(0, number_of_subgraphs)
                diff = number_of_edges - ecount
                
                # 修改这部分条件判断
                if diff >= self.number_of_pattern_edges:
                    # 增加pattern添加的概率检查
                    should_add_pattern = (subgraphs[sg1]["pattern_valid"] and 
                                        np.random.rand() < alpha and
                                        len(graph_vertex_label_mapping_reversed) > 0)
                    
                    if should_add_pattern:
                        new_ecount = self.add_pattern(subgraphs[sg1], graph_vertex_label_mapping_reversed, graph_edge_label_mapping)
                        if new_ecount == 0:
                            # 如果添加pattern失败，改为添加普通边
                            new_ecount = self.add_edges(subgraphs[sg1], subgraphs[sg2],
                                graph_edge_label_mapping, number_of_edge_labels, self.number_of_pattern_edges)
                    else:
                        new_ecount = self.add_edges(subgraphs[sg1], subgraphs[sg2],
                            graph_edge_label_mapping, number_of_edge_labels, self.number_of_pattern_edges)
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
            
            self.update_subgraphs(subgraphs, graph_edge_label_mapping)
            graph, graph_vertex_mapping, graph_vertex_mapping_reversed = self.merge_subgraphs(subgraphs, graph_edge_label_mapping)
            
            # 准备metadata
            if return_subisomorphisms:
                subisomorphisms = list()
                for sg, subgraph in enumerate(subgraphs):
                    for subisomorphism in self.pattern_checker.get_subisomorphisms(subgraph, self.pattern):
                        # 确保映射正确
                        try:
                            mapped_subiso = [graph_vertex_mapping_reversed[(sg, v)] for v in subisomorphism]
                            subisomorphisms.append(mapped_subiso)
                        except KeyError as e:
                            print(f"映射错误: {e}")
                            continue
                scount = len(subisomorphisms)
                metadata = {"counts": scount, "subisomorphisms": subisomorphisms}
            else:
                counts = 0
                for subgraph in subgraphs:
                    counts += self.pattern_checker.count_subisomorphisms(subgraph, self.pattern)
                metadata = {"counts": counts, "subisomorphisms": []}
            
            # 在最终合并的图上验证pattern计数，确保准确性
            final_count = self.pattern_checker.count_subisomorphisms(graph, self.pattern)
            # 如果差异较大，使用更准确的计数
            if abs(final_count - metadata["counts"]) > 1:  # 允许小的差异
                print(f"计数校正: 子图计数={metadata['counts']}, 合并图计数={final_count}")
                metadata["counts"] = final_count
                metadata["verified_count"] = final_count
            
            # 确保图属性是字符型
            graph.vs["label"] = [str(x) for x in graph.vs["label"]]
            graph.es["label"] = [str(x) for x in graph.es["label"]]  # 字符型边标签
            
            result = (graph, metadata)

            if metadata["counts"] > max_pattern_counts:
                alpha = alpha * max_pattern_counts / metadata["counts"] * DECAY
                result = self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                    alpha=alpha, max_subgraph=max_subgraph, max_pattern_counts=max_pattern_counts,
                    return_subisomorphisms=return_subisomorphisms)
                self._recursion_depth = 0

            return result
        

    # def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
    #     sg = subgraph["sg"]
    #     subisomorphism = list()
    #     for vertex_label in self.pattern.vs["label"]:
    #         subisomorphism.append(sample_element(graph_vertex_label_mapping_reversed[(sg, vertex_label)]))
    #     new_ecount = 0
    #     for (pattern_u, pattern_v), pattern_edge_labels in self.pattern_edge_label_mapping.items():
    #         graph_u = subisomorphism[pattern_u]
    #         graph_v = subisomorphism[pattern_v]
    #         graph_edge_labels = graph_edge_label_mapping[(sg, graph_u, sg, graph_v)]
    #         edge_label_diff = pattern_edge_labels - graph_edge_labels
    #         for edge_label in edge_label_diff:
    #             graph_edge_labels.add(edge_label)
    #         new_ecount += len(edge_label_diff)
    #     return new_ecount
    # def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
    #     sg = subgraph["sg"]
    #     subisomorphism = list()
        
    #     # 确保有足够的节点可以映射到pattern节点
    #     for vertex_label in self.pattern.vs["label"]:
    #         if (sg, vertex_label) not in graph_vertex_label_mapping_reversed or \
    #         len(graph_vertex_label_mapping_reversed[(sg, vertex_label)]) == 0:
    #             # 如果没有该标签的节点，需要创建一个
    #             # 这里需要修改图结构，比较复杂
    #             # 更简单的方法是返回0，让系统生成新的图
    #             return 0
                
    #         subisomorphism.append(sample_element(graph_vertex_label_mapping_reversed[(sg, vertex_label)]))
        
    #     new_ecount = 0
    #     for (pattern_u, pattern_v), pattern_edge_labels in self.pattern_edge_label_mapping.items():
    #         graph_u = subisomorphism[pattern_u]
    #         graph_v = subisomorphism[pattern_v]
    #         graph_edge_labels = graph_edge_label_mapping[(sg, graph_u, sg, graph_v)]
    #         edge_label_diff = pattern_edge_labels - graph_edge_labels
    #         for edge_label in edge_label_diff:
    #             graph_edge_labels.add(edge_label)
    #         new_ecount += len(edge_label_diff)
    #     return new_ecount
    # def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
    #     sg = subgraph["sg"]
    #     new_ecount = 0
        
    #     # 创建pattern节点到子图节点的映射
    #     node_mapping = {}
    #     available_nodes = list(range(subgraph.vcount()))
        
    #     # 为每个pattern节点找到或创建对应的子图节点
    #     for pattern_v in range(self.pattern.vcount()):
    #         pattern_label = self.pattern.vs[pattern_v]["label"]
            
    #         # 检查是否有匹配标签的节点
    #         if (sg, pattern_label) in graph_vertex_label_mapping_reversed:
    #             candidates = graph_vertex_label_mapping_reversed[(sg, pattern_label)]
    #             if candidates:
    #                 graph_node = sample_element(candidates)
    #                 node_mapping[pattern_v] = graph_node
    #                 # 从可用节点中移除
    #                 if graph_node in available_nodes:
    #                     available_nodes.remove(graph_node)
    #                 continue
            
    #         # 如果没有匹配的节点，修改一个可用节点的标签
    #         if available_nodes:
    #             graph_node = available_nodes.pop()
    #             old_label = subgraph.vs[graph_node]["label"]
                
    #             # 更新子图节点标签
    #             subgraph.vs[graph_node]["label"] = pattern_label
                
    #             # 更新反向映射
    #             if (sg, old_label) in graph_vertex_label_mapping_reversed:
    #                 if graph_node in graph_vertex_label_mapping_reversed[(sg, old_label)]:
    #                     graph_vertex_label_mapping_reversed[(sg, old_label)].remove(graph_node)
                
    #             if (sg, pattern_label) not in graph_vertex_label_mapping_reversed:
    #                 graph_vertex_label_mapping_reversed[(sg, pattern_label)] = []
    #             graph_vertex_label_mapping_reversed[(sg, pattern_label)].append(graph_node)
                
    #             node_mapping[pattern_v] = graph_node
    #         else:
    #             # 没有可用节点，无法添加pattern
    #             return 0
        
    #     # 添加pattern的边
    #     for pattern_edge in self.pattern.es:
    #         pattern_u = pattern_edge.source
    #         pattern_v = pattern_edge.target
    #         edge_label = pattern_edge["label"]
            
    #         graph_u = node_mapping[pattern_u]
    #         graph_v = node_mapping[pattern_v]
            
    #         # 创建边的键
    #         edge_key = (sg, graph_u, sg, graph_v)
            
    #         # 确保边键在映射中
    #         if edge_key not in graph_edge_label_mapping:
    #             graph_edge_label_mapping[edge_key] = set()
            
    #         # 添加边标签
    #         if edge_label not in graph_edge_label_mapping[edge_key]:
    #             graph_edge_label_mapping[edge_key].add(edge_label)
    #             new_ecount += 1
        
    #     return new_ecount

    # def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
    #     sg = subgraph["sg"]
    #     new_ecount = 0
        
    #     # 检查子图是否有足够的节点来匹配pattern
    #     if subgraph.vcount() < self.pattern.vcount():
    #         # 添加缺失的节点
    #         nodes_to_add = self.pattern.vcount() - subgraph.vcount()
    #         subgraph.add_vertices(nodes_to_add)
    #         # 为新添加的节点分配标签
    #         for i in range(nodes_to_add):
    #             # 可以使用pattern中的标签或者默认标签
    #             new_node_index = subgraph.vcount() - nodes_to_add + i
    #             pattern_label = self.pattern.vs[new_node_index]["label"]
    #             subgraph.vs[new_node_index]["label"] = pattern_label
    #             # 更新标签映射
    #             if (sg, pattern_label) not in graph_vertex_label_mapping_reversed:
    #                 graph_vertex_label_mapping_reversed[(sg, pattern_label)] = []
    #             graph_vertex_label_mapping_reversed[(sg, pattern_label)].append(new_node_index)
        
    #     # 创建pattern节点到子图节点的映射
    #     node_mapping = {}
    #     available_nodes = list(range(subgraph.vcount()))
        
    #     # 为每个pattern节点找到或创建对应的子图节点
    #     for pattern_v in range(self.pattern.vcount()):
    #         pattern_label = self.pattern.vs[pattern_v]["label"]
            
    #         # 检查是否有匹配标签的节点
    #         if (sg, pattern_label) in graph_vertex_label_mapping_reversed:
    #             candidates = graph_vertex_label_mapping_reversed[(sg, pattern_label)]
    #             if candidates:
    #                 graph_node = sample_element(candidates)
    #                 node_mapping[pattern_v] = graph_node
    #                 # 从可用节点中移除
    #                 if graph_node in available_nodes:
    #                     available_nodes.remove(graph_node)
    #                 continue
            
    #         # 如果没有匹配的节点，修改一个可用节点的标签
    #         if available_nodes:
    #             graph_node = available_nodes.pop()
    #             old_label = subgraph.vs[graph_node]["label"]
                
    #             # 更新子图节点标签
    #             subgraph.vs[graph_node]["label"] = pattern_label
                
    #             # 更新反向映射
    #             if (sg, old_label) in graph_vertex_label_mapping_reversed:
    #                 if graph_node in graph_vertex_label_mapping_reversed[(sg, old_label)]:
    #                     graph_vertex_label_mapping_reversed[(sg, old_label)].remove(graph_node)
                
    #             if (sg, pattern_label) not in graph_vertex_label_mapping_reversed:
    #                 graph_vertex_label_mapping_reversed[(sg, pattern_label)] = []
    #             graph_vertex_label_mapping_reversed[(sg, pattern_label)].append(graph_node)
                
    #             node_mapping[pattern_v] = graph_node
    #         else:
    #             # 没有可用节点，无法添加pattern
    #             return 0
        
    #     # 添加pattern的边
    #     for pattern_edge in self.pattern.es:
    #         pattern_u = pattern_edge.source
    #         pattern_v = pattern_edge.target
    #         edge_label = pattern_edge["label"]
            
    #         graph_u = node_mapping[pattern_u]
    #         graph_v = node_mapping[pattern_v]
            
    #         # 创建边的键
    #         edge_key = (sg, graph_u, sg, graph_v)
            
    #         # 确保边键在映射中
    #         if edge_key not in graph_edge_label_mapping:
    #             graph_edge_label_mapping[edge_key] = set()
            
    #         # 添加边标签
    #         if edge_label not in graph_edge_label_mapping[edge_key]:
    #             graph_edge_label_mapping[edge_key].add(edge_label)
    #             new_ecount += 1
        
    #     return new_ecount

    def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
        sg = subgraph["sg"]
        new_ecount = 0
        
        # 检查子图是否有足够的节点来匹配pattern
        if subgraph.vcount() < self.pattern.vcount():
            # 添加缺失的节点
            nodes_to_add = self.pattern.vcount() - subgraph.vcount()
            subgraph.add_vertices(nodes_to_add)
            # 为新添加的节点初始化标签
            for i in range(nodes_to_add):
                new_node_index = subgraph.vcount() - nodes_to_add + i
                subgraph.vs[new_node_index]["label"] = "0"  # 默认标签
        
        # 创建pattern节点到子图节点的映射
        node_mapping = {}
        available_nodes = list(range(subgraph.vcount()))
        np.random.shuffle(available_nodes)  # 随机打乱可用节点
        
        # 创建一个pattern标签的副本用于分配
        pattern_labels = list(self.pattern.vs["label"])
        np.random.shuffle(pattern_labels)  # 随机打乱pattern标签
        
        # 首先尝试保留一些原始标签，然后分配pattern标签
        # 确保所有pattern标签都被分配到子图节点上
        pattern_label_counter = Counter(self.pattern.vs["label"])
        subgraph_label_counter = Counter(subgraph.vs["label"])
        
        # 更新标签计数，确保子图有足够的pattern标签
        for label, needed_count in pattern_label_counter.items():
            current_count = subgraph_label_counter[label]
            if current_count < needed_count:
                # 需要更多该标签的节点
                missing_count = needed_count - current_count
                # 从可用节点中选择一些节点来修改标签
                nodes_to_modify = available_nodes[:missing_count]
                available_nodes = available_nodes[missing_count:]
                
                for node_idx in nodes_to_modify:
                    old_label = subgraph.vs[node_idx]["label"]
                    subgraph.vs[node_idx]["label"] = label
                    
                    # 更新反向映射
                    if (sg, old_label) in graph_vertex_label_mapping_reversed:
                        if node_idx in graph_vertex_label_mapping_reversed[(sg, old_label)]:
                            graph_vertex_label_mapping_reversed[(sg, old_label)].remove(node_idx)
                    
                    if (sg, label) not in graph_vertex_label_mapping_reversed:
                        graph_vertex_label_mapping_reversed[(sg, label)] = []
                    graph_vertex_label_mapping_reversed[(sg, label)].append(node_idx)
                
                # 更新计数
                subgraph_label_counter[label] += missing_count
        
        # 现在建立节点映射关系
        used_nodes = set()
        for pattern_v in range(self.pattern.vcount()):
            pattern_label = self.pattern.vs[pattern_v]["label"]
            
            # 从具有匹配标签且未使用的节点中选择一个
            candidates = []
            if (sg, pattern_label) in graph_vertex_label_mapping_reversed:
                candidates = [node for node in graph_vertex_label_mapping_reversed[(sg, pattern_label)] 
                             if node not in used_nodes]
            
            if candidates:
                graph_node = sample_element(candidates)
                node_mapping[pattern_v] = graph_node
                used_nodes.add(graph_node)
            else:
                # 应该不会发生，因为我们已经确保有足够的标签
                if available_nodes:
                    graph_node = available_nodes.pop()
                    node_mapping[pattern_v] = graph_node
                    used_nodes.add(graph_node)
                else:
                    return 0  # 极端情况下无法添加pattern
        
        # 添加pattern的边
        for pattern_edge in self.pattern.es:
            pattern_u = pattern_edge.source
            pattern_v = pattern_edge.target
            edge_label = pattern_edge["label"]
            
            graph_u = node_mapping[pattern_u]
            graph_v = node_mapping[pattern_v]
            
            # 创建边的键
            edge_key = (sg, graph_u, sg, graph_v)
            
            # 确保边键在映射中
            if edge_key not in graph_edge_label_mapping:
                graph_edge_label_mapping[edge_key] = set()
            
            # 添加边标签
            if edge_label not in graph_edge_label_mapping[edge_key]:
                graph_edge_label_mapping[edge_key].add(edge_label)
                new_ecount += 1
        
        return new_ecount

    # def add_edges(self, subgraph1, subgraph2, graph_edge_label_mapping, graph_number_of_edge_labels, number_of_edges):
    #     sg1 = subgraph1["sg"]
    #     sg2 = subgraph2["sg"]
    #     g1_vcount = subgraph1.vcount()
    #     g2_vcount = subgraph2.vcount()
    #     new_ecount = 0
    #     invalid_cnt = 0
        
    #     while invalid_cnt < 10 and new_ecount < number_of_edges:
    #         v1 = np.random.randint(0, g1_vcount)
    #         v2 = np.random.randint(0, g2_vcount)
            
    #         # 避免创建自环边
    #         if sg1 == sg2 and v1 == v2:
    #             invalid_cnt += 1
    #             continue
            
    #         # 获取节点标签（器件编码），保持字符串类型
    #         source_label = subgraph1.vs[v1]["label"]
    #         target_label = subgraph2.vs[v2]["label"]
            
    #         # 生成边标签，确保类型处理正确
    #         try:
    #             # 尝试使用数字标签生成边标签
    #             if source_label.isdigit() and target_label.isdigit():
    #                 if get_direction():
    #                     edge_label = generate_edge_label_by_device_types(int(source_label), int(target_label))
    #                     y = (sg1, v1, sg2, v2)
    #                 else:
    #                     edge_label = generate_edge_label_by_device_types(int(target_label), int(source_label))
    #                     y = (sg2, v2, sg1, v1)
    #             else:
    #                 # 如果标签不是数字，从pattern边标签中随机选择
    #                 edge_label = np.random.choice(self.pattern_edge_labels) if self.pattern_edge_labels else "0"
    #                 y = (sg1, v1, sg2, v2)
    #         except Exception as e:
    #             # 出现异常时使用默认标签
    #             edge_label = "0"
    #             y = (sg1, v1, sg2, v2)
                    
    #         graph_edge_labels = graph_edge_label_mapping[y]
    #         if edge_label in graph_edge_labels:
    #             invalid_cnt += 1
    #             continue
                
    #         graph_edge_labels.add(edge_label)
    #         new_ecount += 1
    #         invalid_cnt = 0
            
    #     # 返回新增的边数
    #     return new_ecount
    # def add_edges(self, subgraph1, subgraph2, graph_edge_label_mapping, graph_number_of_edge_labels, number_of_edges):
    #     sg1 = subgraph1["sg"]
    #     sg2 = subgraph2["sg"]
    #     g1_vcount = subgraph1.vcount()
    #     g2_vcount = subgraph2.vcount()
    #     new_ecount = 0
    #     invalid_cnt = 0
        
    #     while invalid_cnt < 10 and new_ecount < number_of_edges:
    #         v1 = np.random.randint(0, g1_vcount)
    #         v2 = np.random.randint(0, g2_vcount)
            
    #         # 避免创建自环边
    #         if sg1 == sg2 and v1 == v2:
    #             invalid_cnt += 1
    #             continue
            
    #         # 获取节点标签（器件编码），保持字符串类型
    #         source_label = subgraph1.vs[v1]["label"]
    #         target_label = subgraph2.vs[v2]["label"]
            
    #         # 生成边标签，确保类型处理正确
    #         try:
    #             # 尝试使用数字标签生成边标签
    #             if source_label.isdigit() and target_label.isdigit():
    #                 edge_label = generate_edge_label_by_device_types(int(source_label), int(target_label))
                    
    #                 # 创建反向边标签，通过交换out_to_in和in_to_out部分
    #                 # edge_label格式: out_to_in(3位) + in_to_out(3位) + size_relation(2位)
    #                 if len(edge_label) >= 8:
    #                     out_to_in = edge_label[:3]
    #                     in_to_out = edge_label[3:6]
    #                     size_relation = edge_label[6:8]
    #                     # 对于size_relation，如果是'09'则变为'90'，如果是'90'则变为'09'，'99'保持不变
    #                     if size_relation == '09':
    #                         reversed_size_relation = '90'
    #                     elif size_relation == '90':
    #                         reversed_size_relation = '09'
    #                     else:  # '99'或其他情况保持不变
    #                         reversed_size_relation = size_relation
    #                     reverse_edge_label = in_to_out + out_to_in + reversed_size_relation
    #                 else:
    #                     # 如果标签长度不正确，使用默认方式处理
    #                     reverse_edge_label = generate_edge_label_by_device_types(int(target_label), int(source_label))
                    
    #                 # 添加正向边
    #                 y_forward = (sg1, v1, sg2, v2)
    #                 graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
    #                 if edge_label not in graph_edge_labels_forward:
    #                     graph_edge_labels_forward.add(edge_label)
    #                     new_ecount += 1
                        
    #                     # 添加反向边
    #                     y_reverse = (sg2, v2, sg1, v1)
    #                     graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
    #                     graph_edge_labels_reverse.add(reverse_edge_label)
    #                     new_ecount += 1
    #             else:
    #                 # 如果标签不是数字，从pattern边标签中随机选择
    #                 edge_label = np.random.choice(self.pattern_edge_labels) if self.pattern_edge_labels else "0"
                    
    #                 # 添加正向边
    #                 y_forward = (sg1, v1, sg2, v2)
    #                 graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
    #                 if edge_label not in graph_edge_labels_forward:
    #                     graph_edge_labels_forward.add(edge_label)
    #                     new_ecount += 1
                        
    #                     # 添加反向边（使用相同标签）
    #                     y_reverse = (sg2, v2, sg1, v1)
    #                     graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
    #                     graph_edge_labels_reverse.add(edge_label)
    #                     new_ecount += 1
    #         except Exception as e:
    #             # 出现异常时使用默认标签
    #             edge_label = "0"
                
    #             # 添加正向边
    #             y_forward = (sg1, v1, sg2, v2)
    #             graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
    #             if edge_label not in graph_edge_labels_forward:
    #                 graph_edge_labels_forward.add(edge_label)
    #                 new_ecount += 1
                    
    #                 # 添加反向边
    #                 y_reverse = (sg2, v2, sg1, v1)
    #                 graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
    #                 graph_edge_labels_reverse.add(edge_label)
    #                 new_ecount += 1
                    
    #         invalid_cnt = 0
            
    #     # 返回新增的边数
    #     return new_ecount
    def add_edges(self, subgraph1, subgraph2, graph_edge_label_mapping, graph_number_of_edge_labels, number_of_edges):
        sg1 = subgraph1["sg"]
        sg2 = subgraph2["sg"]
        g1_vcount = subgraph1.vcount()
        g2_vcount = subgraph2.vcount()
        new_ecount = 0
        invalid_cnt = 0
        
        while invalid_cnt < 10 and new_ecount < number_of_edges:
            v1 = np.random.randint(0, g1_vcount)
            v2 = np.random.randint(0, g2_vcount)
            
            # 避免创建自环边
            if sg1 == sg2 and v1 == v2:
                invalid_cnt += 1
                continue
            
            # 获取节点标签（器件编码），保持字符串类型
            source_label = subgraph1.vs[v1]["label"]
            target_label = subgraph2.vs[v2]["label"]
            
            # 生成边标签，确保类型处理正确
            try:
                # 尝试使用数字标签生成边标签
                if source_label.isdigit() and target_label.isdigit():
                    edge_label = generate_edge_label_by_device_types(int(source_label), int(target_label))
                    
                    # 创建反向边标签，通过交换out_to_in和in_to_out部分
                    # edge_label格式: out_to_in(3位) + in_to_out(3位) + size_relation(2位)
                    if len(edge_label) >= 8:
                        out_to_in = edge_label[:3]
                        in_to_out = edge_label[3:6]
                        size_relation = edge_label[6:8]
                        # 对于size_relation，如果是'09'则变为'90'，如果是'90'则变为'09'，'99'保持不变
                        if size_relation == '09':
                            reversed_size_relation = '90'
                        elif size_relation == '90':
                            reversed_size_relation = '09'
                        else:  # '99'或其他情况保持不变
                            reversed_size_relation = size_relation
                        reverse_edge_label = in_to_out + out_to_in + reversed_size_relation
                    else:
                        # 如果标签长度不正确，使用默认方式处理
                        reverse_edge_label = generate_edge_label_by_device_types(int(target_label), int(source_label))
                    
                    # 添加正向边
                    y_forward = (sg1, v1, sg2, v2)
                    graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
                    
                    # 检查是否已存在该标签
                    if edge_label not in graph_edge_labels_forward:
                        graph_edge_labels_forward.add(edge_label)
                        new_ecount += 1
                    else:
                        invalid_cnt += 1
                        continue
                    
                    # 添加反向边（仅当不是自环边时）
                    if sg1 != sg2 or v1 != v2:
                        y_reverse = (sg2, v2, sg1, v1)
                        graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
                        if reverse_edge_label not in graph_edge_labels_reverse:
                            graph_edge_labels_reverse.add(reverse_edge_label)
                            new_ecount += 1
                    
                else:
                    # 如果标签不是数字，从pattern边标签中随机选择
                    edge_label = np.random.choice(self.pattern_edge_label_list) if hasattr(self, 'pattern_edge_label_list') and len(self.pattern_edge_label_list) > 0 else "00100199"
                    
                    # 添加正向边
                    y_forward = (sg1, v1, sg2, v2)
                    graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
                    
                    # 检查是否已存在该标签
                    if edge_label not in graph_edge_labels_forward:
                        graph_edge_labels_forward.add(edge_label)
                        new_ecount += 1
                    else:
                        invalid_cnt += 1
                        continue
                    
                    # 添加反向边（使用相同标签，仅当不是自环边时）
                    if sg1 != sg2 or v1 != v2:
                        y_reverse = (sg2, v2, sg1, v1)
                        graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
                        if edge_label not in graph_edge_labels_reverse:
                            graph_edge_labels_reverse.add(edge_label)
                            new_ecount += 1
                    
            except Exception as e:
                # 出现异常时使用默认标签
                edge_label = "00100199"
                
                # 添加正向边
                y_forward = (sg1, v1, sg2, v2)
                graph_edge_labels_forward = graph_edge_label_mapping[y_forward]
                if edge_label not in graph_edge_labels_forward:
                    graph_edge_labels_forward.add(edge_label)
                    new_ecount += 1
                else:
                    invalid_cnt += 1
                    continue
                
                # 添加反向边
                if sg1 != sg2 or v1 != v2:
                    y_reverse = (sg2, v2, sg1, v1)
                    graph_edge_labels_reverse = graph_edge_label_mapping[y_reverse]
                    if edge_label not in graph_edge_labels_reverse:
                        graph_edge_labels_reverse.add(edge_label)
                        new_ecount += 1
                    
            invalid_cnt = 0
            
        # 返回新增的边数
        return new_ecount


# def generate_graphs(pattern, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels, \
#     alpha, max_pattern_counts, max_subgraph, return_subisomorphisms, number_of_graphs):
#     graph_generator = GraphGenerator(pattern)
#     results = list()
#     for g in range(number_of_graphs):
#         graph, metadata = graph_generator.generate(
#                 number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
#                 alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
#                 return_subisomorphisms=return_subisomorphisms)
#         print("%d/%d" % (g+1, number_of_graphs), "number of subisomorphisms: %d" % (metadata["counts"]))
#         results.append((graph, metadata))
#     return results
def generate_graphs(graph_generator, number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
    alpha, max_pattern_counts, max_subgraph, number_of_graphs, save_graph_dir, save_metadata_dir):
    
    graphs_id = "G_N%d_E%d_NL%d_EL%d_A%.2f" % (
        number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels, alpha)
    
    # 获取pattern名称，用于记录子图信息
    pattern_name = "unknown_pattern"
    if hasattr(graph_generator, 'pattern') and hasattr(graph_generator.pattern, 'name'):
        pattern_name = graph_generator.pattern.name
    else:
        # 从保存目录路径中提取pattern名称
        pattern_name = os.path.basename(save_graph_dir)
    
    # 子图信息记录文件
    subgraph_info_file = os.path.join(save_metadata_dir, f"subgraph_info_{graphs_id}.txt")
    subgraph_info_lines = []
    
    graphs = []
    metadatas = []
    
    for g in range(number_of_graphs):
        graph, metadata = graph_generator.generate(
            number_of_graph_vertices, number_of_graph_edges, 
            number_of_graph_vertex_labels, number_of_graph_edge_labels,
            alpha, max_pattern_counts=max_pattern_counts, 
            max_subgraph=max_subgraph, return_subisomorphisms=True)
        
        # 确保图属性是GML兼容类型
        graph.vs["label"] = [str(x) for x in graph.vs["label"]]
        graph.es["label"] = [str(x) for x in graph.es["label"]]  # 已经是字符型
        
        # 将生成的图和元数据添加到列表中（这是之前缺失的关键步骤）
        graphs.append(graph)
        metadatas.append(metadata)
        
        # 验证metadata的准确性
        subisomorphisms = metadata.get("subisomorphisms", [])
        counts = metadata.get("counts", 0)
        
        # 如果计数为0但应该有pattern，记录警告
        if counts == 0 and alpha > 0:
            print(f"警告: 图 {graphs_id}_{g} 没有检测到pattern，但alpha={alpha}")
        
        # 双重验证：使用pattern_checker重新计数
        pattern_checker = PatternChecker()
        actual_count = pattern_checker.count_subisomorphisms(graph, graph_generator.pattern)
        
        if actual_count != counts:
            print(f"警告: 图 {graphs_id}_{g} 的pattern计数不匹配: "
                  f"metadata={counts}, 实际={actual_count}")
            # 更新metadata
            metadata["counts"] = actual_count
            metadata["verified_count"] = actual_count
            metadatas[g] = metadata  # 更新列表中的metadata
            
        # 记录子图信息
        if actual_count > 0:
            subgraph_info_lines.append(f"Graph: {graphs_id}_{g}.gml, Subgraph Count: {actual_count}")
            for i, subiso in enumerate(subisomorphisms):
                subgraph_info_lines.append(f"  Subgraph {i+1}: Nodes {subiso}")
        else:
            subgraph_info_lines.append(f"Graph: {graphs_id}_{g}.gml, Subgraph Count: 0")
    
    # 写入子图信息到txt文件
    if subgraph_info_lines:
        try:
            os.makedirs(os.path.dirname(subgraph_info_file), exist_ok=True)
            with open(subgraph_info_file, "w", encoding="utf-8") as f:
                f.write(f"Pattern: {pattern_name}\n")
                f.write(f"Configuration: {graphs_id}\n")
                f.write("=" * 50 + "\n")
                f.write("\n".join(subgraph_info_lines))
                f.write("\n")
        except Exception as e:
            print(f"Error writing subgraph info file {subgraph_info_file}: {e}")
    
    # 在单个进程中执行所有文件写入操作，避免多进程文件操作问题
    for g, (graph, metadata) in enumerate(zip(graphs, metadatas)):
        # 确保目录存在
        os.makedirs(save_graph_dir, exist_ok=True)
        os.makedirs(save_metadata_dir, exist_ok=True)
        
        graph_filename = os.path.join(save_graph_dir, graphs_id + "_%d.gml" % (g))
        meta_filename = os.path.join(save_metadata_dir, graphs_id + "_%d.meta" % (g))
        
        # 使用更安全的文件写入方法
        try:
            graph.write(graph_filename)
        except Exception as e:
            print(f"Error writing graph file {graph_filename}: {e}")
            # 尝试替代方法
            try:
                with open(graph_filename, 'w', encoding='utf-8') as f:
                    graph.write(f)
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                # 创建一个简单的错误指示文件
                with open(graph_filename.replace('.gml', '_error.txt'), 'w') as f:
                    f.write(f"Failed to write graph due to: {e2}\n")
                    f.write(f"Graph has {graph.vcount()} vertices and {graph.ecount()} edges\n")
        
        try:
            with open(meta_filename, "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Error writing metadata file {meta_filename}: {e}")
    
    return graphs_id

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
    parser.add_argument("--number_of_edges", type=int, default=2048 * 4)
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
        # 确保pattern的边标签是字符型
        pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
        pattern.es["label"] = [str(x) for x in pattern.es["label"]]
        if "key" in pattern.es.attributes():
            pattern.es["key"] = [str(x) for x in pattern.es["key"]]
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