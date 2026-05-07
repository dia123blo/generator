import igraph as ig
import numpy as np
import argparse
import os
import math
import json
import shutil
from collections import Counter, defaultdict
from utils import generate_labels, generate_tree, get_direction, powerset, sample_element, str2bool, retrieve_multiple_edges
from pattern_checker import PatternChecker
from time import time
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool

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
    def __init__(self, patterns):
        self.patterns = patterns if patterns else [ig.Graph(directed=True)]
        self.number_of_patterns = len(self.patterns)
        self.pattern_checkers = [PatternChecker() for _ in self.patterns]
        
        # 存储每个模板的元数据
        self.pattern_meta = []
        for pattern in self.patterns:
            meta = {
                "number_of_vertices": pattern.vcount(),
                "number_of_edges": pattern.ecount(),
                "vertex_label_counter": Counter(pattern.vs["label"]),
                "edge_label_counter": Counter(pattern.es["label"]),
                "edge_label_mapping": defaultdict(set),
                "vertex_edge_label_mapping": defaultdict(set)
            }
            for edge in pattern.es:
                meta["edge_label_mapping"][edge.tuple].add(edge["label"])
                key = (pattern.vs[edge.source]["label"], pattern.vs[edge.target]["label"])
                meta["vertex_edge_label_mapping"][key].add(edge["label"])
            self.pattern_meta.append(meta)
        # self.number_of_pattern_vertices = pattern.vcount()
        # self.number_of_pattern_edges = pattern.ecount()
        # self.pattern_vertex_label_counter = Counter(pattern.vs["label"])
        # self.pattern_edge_label_counter = Counter(pattern.es["label"])

        # self.pattern_edge_label_mapping = defaultdict(set)
        # self.pattern_vertex_edge_label_mapping = defaultdict(set)
        # for edge in pattern.es:
        #     self.pattern_edge_label_mapping[edge.tuple].add(edge["label"])
        #     key = (pattern.vs[edge.source]["label"], pattern.vs[edge.target]["label"])
        #     self.pattern_vertex_edge_label_mapping[key].add(edge["label"])

        # self.number_of_pattern_vertex_labels = int(max(pattern.vs["label"])) + 1
        # self.number_of_pattern_edge_labels = int(max(pattern.es["label"])) + 1

        # self.pattern_nec_tree = self.rewrite_to_nec_tree()
        # self.pattern_nec_tree_vertex_edge_label_mapping = defaultdict(set)
        # for edge in self.pattern_nec_tree.tree.es:
        #     key = (self.pattern_nec_tree.tree.vs[edge.source]["label"], self.pattern_nec_tree.tree.vs[edge.target]["label"])
        #     self.pattern_nec_tree_vertex_edge_label_mapping[key].add(edge["label"])

        # self.pattern_checker = PatternChecker()


    def _add_patterns_to_graph(self, graph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
        print("\n===== 开始模板嵌入 =====")
        total_new_edges = 0
        all_subisomorphisms = []
        
        for pattern_idx, pattern in enumerate(self.patterns):
            print(f"\n处理模板 {pattern_idx}:")
            print(f"  模板顶点: {pattern.vs['label']}")
            print(f"  模板边: {[(e.source, e.target, e['label']) for e in pattern.es]}")
            
            # 尝试创建嵌入
            subisomorphism = []
            for vertex_label in pattern.vs["label"]:
                candidates = graph_vertex_label_mapping_reversed.get(vertex_label, [])
                print(f"  标签 '{vertex_label}' 的候选顶点: {len(candidates)}")
                
                if candidates:
                    selected = np.random.choice(candidates)
                    subisomorphism.append(selected)
                    print(f"    选择顶点 {selected} (标签={graph.vs[selected]['label']})")
                else:
                    print(f"    警告: 没有找到标签 '{vertex_label}' 的候选顶点")
            
            if len(subisomorphism) == len(pattern.vs):
                print(f"  成功创建嵌入: {subisomorphism}")
                new_edges = 0
                
                for edge in pattern.es:
                    src = subisomorphism[edge.source]
                    tgt = subisomorphism[edge.target]
                    key = (src, tgt)
                    
                    # 检查边是否已存在
                    existing_edges = graph_edge_label_mapping.get(key, set())
                    if edge["label"] in existing_edges:
                        print(f"    边 {src}->{tgt} (标签={edge['label']}) 已存在")
                    else:
                        print(f"    添加边 {src}->{tgt} (标签={edge['label']})")
                        existing_edges.add(edge["label"])
                        graph_edge_label_mapping[key] = existing_edges
                        new_edges += 1
                
                print(f"  为模板 {pattern_idx} 添加了 {new_edges} 条新边")
                total_new_edges += new_edges
                all_subisomorphisms.append((pattern_idx, subisomorphism))
            else:
                print(f"  警告: 无法为模板 {pattern_idx} 创建完整嵌入")
        
        print(f"===== 模板嵌入完成，添加了 {total_new_edges} 条边 =====")
        return total_new_edges, all_subisomorphisms
    

    def choose_start_q_vertex(self):
        vs = list()
        for v in self.pattern.vs:
            freq = self.pattern_vertex_label_counter[v["label"]]
            # freq = self.graph_vertex_label_counter[v["label"]]
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
                    # 0 -> 1
                    out_degrees[edge[0]] += 1
                    in_degrees[edge[1]] += 1
            # a clique requires all vertices have the same in degrees and out degrees
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
        # each vertex in the same group has the same label
        # so we do not need to care the vertex label here
        groups_by_adj = defaultdict(list) # key: adj, value: vertices
        for v in group:
            adj = list() # [(mode, e_label, v_id), ...]
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
        
        
        # NECs with same adj
        singleton_group_mapping = dict() # key: v_id, value: adj
        necs = list()
        for adj, vs in groups_by_adj.items():
            if len(vs) > 1:
                necs.append(NEC(sorted(vs), adj=adj, is_clique=False))
            else:
                singleton_group_mapping[vs[0]] = adj
        
        # NECs with cliques and same adj-N_q
        # firstly check the indegree and outdegree
        groups_by_degree = defaultdict(list)
        for v, adj in singleton_group_mapping.items():
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
        
        root = NEC(data=[start_v], adj=adj, vertex_label=self.pattern.vs[start_v]["label"], is_clique=False, nec_id=0)
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

    def update_subgraphs(self, subgraphs, graph_edge_label_mapping):
        new_edges_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_keys_in_subgraphs = [list() for i in range(len(subgraphs))]
        new_edge_labels_in_subgraphs = [list() for i in range(len(subgraphs))]
        subgraphs_vlabels = [subgraph.vs["label"] for subgraph in subgraphs]
        
        for (sg1, v1, sg2, v2), edge_labels in graph_edge_label_mapping.items():
            if sg1 == sg2:
                src_tgt = (v1, v2)
                key = (subgraphs_vlabels[sg1][v1], subgraphs_vlabels[sg2][v2])
                
                # 使用新的 pattern_meta 元数据
                pattern_edge_labels = set()
                for meta in self.pattern_meta:
                    # 从每个模板的元数据中获取边标签集合
                    if key in meta["vertex_edge_label_mapping"]:
                        pattern_edge_labels.update(meta["vertex_edge_label_mapping"][key])
                
                # 过滤掉不在任何模板中的边标签
                edge_labels = [edge_label for edge_label in edge_labels if edge_label in pattern_edge_labels] 
                new_edges_in_subgraphs[sg1].extend([src_tgt] * len(edge_labels))
                new_edge_keys_in_subgraphs[sg1].extend(range(len(edge_labels)))
                new_edge_labels_in_subgraphs[sg1].extend(edge_labels)
        
        for sg, subgraph in enumerate(subgraphs):
            subgraph.add_edges(new_edges_in_subgraphs[sg])
            subgraph.es["label"] = new_edge_labels_in_subgraphs[sg]
            subgraph.es["key"] = new_edge_keys_in_subgraphs[sg]

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

    def generate(self, number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
        alpha, max_pattern_counts=-1, max_subgraph=512, return_subisomorphisms=False):

        
        print(f"\n生成图: 顶点={number_of_vertices}, 边={number_of_edges}, 顶点标签={number_of_vertex_labels}, 边标签={number_of_edge_labels}")
        required_vertex_labels = max(len(meta["vertex_label_counter"]) for meta in self.pattern_meta)
        required_edge_labels = max(len(meta["edge_label_counter"]) for meta in self.pattern_meta)
        if number_of_vertex_labels < required_vertex_labels:
            print(f"调整顶点标签数: {number_of_vertex_labels} -> {required_vertex_labels}")
            number_of_vertex_labels = required_vertex_labels
        
        if number_of_edge_labels < required_edge_labels:
            print(f"调整边标签数: {number_of_edge_labels} -> {required_edge_labels}")
            number_of_edge_labels = required_edge_labels
        
        graph_pattern_valid = True

        for meta in self.pattern_meta:

            print(f"检查模板要求: 顶点标签={len(meta['vertex_label_counter'])}, 边标签={len(meta['edge_label_counter'])}")
 
            # 检查顶点标签
            if number_of_vertex_labels < len(meta["vertex_label_counter"]):

                print(f"顶点标签不足: 图有 {number_of_vertex_labels}，模板需要 {len(meta['vertex_label_counter'])}")
 
                graph_pattern_valid = False
                break
            # 检查边标签
            if number_of_edge_labels < len(meta["edge_label_counter"]):

                print(f"边标签不足: 图有 {number_of_edge_labels}，模板需要 {len(meta['edge_label_counter'])}")
 
                graph_pattern_valid = False
                break


     
        # if number_of_vertex_labels < self.number_of_pattern_vertex_labels:
        #     print("WARNING: the number of graph vertex labels (%d) is less than the number of pattern vertex labels (%d)." % (
        #         number_of_vertex_labels, self.number_of_pattern_vertex_labels))
        #     graph_pattern_valid = False
        # if number_of_edge_labels < self.number_of_pattern_edge_labels:
        #     print("WARNING: the number of graph edge labels (%d) is less than the number of pattern edge labels (%d)." % (
        #         number_of_edge_labels, self.number_of_pattern_edge_labels))
        #     graph_pattern_valid = False
            
        if not graph_pattern_valid:
            # no subisomorphism in this setting
            # we can generate the graph randomly
            vertex_labels = generate_labels(number_of_vertices, number_of_vertex_labels)
            edge_labels = generate_labels(number_of_edges, number_of_edge_labels)
            graph = generate_tree(number_of_vertices, directed=True)
            graph_edge_label_mapping = defaultdict(set)
            for e, edge in enumerate(graph.es):
                graph_edge_label_mapping[(0, edge.source, 0, edge.target)].add(edge_labels[e])
            
            ecount = graph.ecount()
            edge_keys = [0] * ecount

            # second, random add edges 
            new_edges = []
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
            graph.es["label"] = edge_labels
            graph.es["key"] = edge_keys
            
            metadata = {"counts": 0, "subisomorphisms": []}
            return graph, metadata
        

        total_pattern_edges = sum(meta["number_of_edges"] for meta in self.pattern_meta)
        if max_pattern_counts != -1 and number_of_edges * alpha > max_pattern_counts * total_pattern_edges:
            alpha = max_pattern_counts * total_pattern_edges / number_of_edges * DECAY
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
            
            subisomorphisms = [] if return_subisomorphisms else None
            ecount = 0
            graph_vertex_label_mapping_reversed = defaultdict(list) # key: (sg, v_label), value: v_ids
            graph_edge_label_mapping = defaultdict(set) # key: (sg1, v1, sg2, v2), value: e_labels
        
            
            for sg in range(number_of_subgraphs):
                # construct a directed tree
                number_of_subgraph_vertices = numbers_of_subgraph_vertices[sg]
                subgraph_vertex_labels = generate_labels(number_of_subgraph_vertices, number_of_vertex_labels)
                subgraph_edge_labels = generate_labels(number_of_subgraph_vertices-1, number_of_edge_labels) # tree label
                subgraph = generate_tree(number_of_subgraph_vertices, directed=True)
                subgraph["sg"] = sg
                subgraph.vs["label"] = subgraph_vertex_labels
                
                ecount += (number_of_subgraph_vertices-1)
                subgraphs.append(subgraph)
                for v_id, v_label in enumerate(subgraph_vertex_labels):
                    graph_vertex_label_mapping_reversed[(sg, v_label)].append(v_id)
                for e, (v1, v2) in enumerate(subgraph.get_edgelist()):
                    graph_edge_label_mapping[(sg, v1, sg, v2)].add(subgraph_edge_labels[e])
                subgraph.delete_edges(None)

                # 检查子图是否满足至少一个模板的顶点标签要求
                subgraph_pattern_valid = True
                subgraph_vertex_label_counter = Counter(subgraph_vertex_labels)
                
                # 修复：使用新的 pattern_meta 元数据
                for meta in self.pattern_meta:
                    for vertex_label, cnt in meta["vertex_label_counter"].items():
                        if subgraph_vertex_label_counter[vertex_label] < cnt:
                            subgraph_pattern_valid = False
                            break
                    if not subgraph_pattern_valid:  # 如果已经无效，可以提前退出
                        break
                
                subgraph["pattern_valid"] = subgraph_pattern_valid
            
            invalid_cnt = 0
            # 计算还需要添加的边数
            diff = number_of_edges - ecount
            
            # 修复：替换对 number_of_pattern_edges 的引用
            # 计算所有模板的平均边数（用于决策）
            avg_pattern_edges = sum(meta["number_of_edges"] for meta in self.pattern_meta) / len(self.pattern_meta)
            
            while invalid_cnt < 10 and ecount < number_of_edges:
                # 计算还需要添加的边数
                diff = number_of_edges - ecount
                
                # 随机选择两个子图索引
                sg1_index = np.random.randint(0, len(subgraphs))
                sg2_index = np.random.randint(0, len(subgraphs))
                
                # 获取实际子图对象
                selected_subgraph1 = subgraphs[sg1_index]
                selected_subgraph2 = subgraphs[sg2_index]
                
                # 使用平均模板边数进行决策
                if any(subgraph["pattern_valid"] for subgraph in subgraphs) and np.random.rand() < alpha:
                    # 如果剩余边数足够添加一个平均大小的模板
                    if diff >= avg_pattern_edges:
                        # 尝试添加所有模板
                        new_ecount, new_subisomorphisms = self._add_patterns_to_graph(
                            selected_subgraph1, graph_vertex_label_mapping_reversed, graph_edge_label_mapping)
                        
                        if new_ecount > 0:
                            ecount += new_ecount
                            invalid_cnt = 0
                            if return_subisomorphisms and new_subisomorphisms:
                                subisomorphisms.extend(new_subisomorphisms)
                        else:
                            invalid_cnt += 1
                    else:
                        # 添加随机边，数量为剩余边数
                        new_ecount = self.add_edges(
                            selected_subgraph1, selected_subgraph2,  # 传递单个子图对象
                            graph_edge_label_mapping, number_of_edge_labels, diff)
                        
                        if new_ecount > 0:
                            ecount += new_ecount
                            invalid_cnt = 0
                        else:
                            invalid_cnt += 1
                else:
                    # 添加随机边
                    new_ecount = self.add_edges(
                        selected_subgraph1, selected_subgraph2,  # 传递单个子图对象
                        graph_edge_label_mapping, number_of_edge_labels, 1)
                    
                    if new_ecount > 0:
                        ecount += new_ecount
                        invalid_cnt = 0
                    else:
                        invalid_cnt += 1
                

            if ecount < number_of_edges:
                alpha = alpha * ecount / number_of_edges * DECAY
                return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                    alpha=alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph,
                    return_subisomorphisms=return_subisomorphisms)
            
            self.update_subgraphs(subgraphs, graph_edge_label_mapping)
            graph, graph_vertex_mapping, graph_vertex_mapping_reversed = self.merge_subgraphs(subgraphs, graph_edge_label_mapping)
            
            # 在生成图的最后部分：
            if return_subisomorphisms:
                detailed_subisomorphisms = []
                total_subisomorphisms = 0  # 添加这个变量来跟踪总数
                for sg, subgraph in enumerate(subgraphs):

                    # total_subisomorphisms += len(matches)
                    subgraph_results = {
                        "subgraph_index": sg,
                        "subgraph_summary": subgraph.summary(),
                        "pattern_matches": []
                    }
                    
                    for pattern_idx, pattern in enumerate(self.patterns):
                        pattern_matches = {
                            "pattern_index": pattern_idx,
                            "pattern_summary": pattern.summary(),
                            "matches": []
                        }
                        
                        matches = self.pattern_checkers[pattern_idx].get_subisomorphisms(subgraph, pattern)
                        total_subisomorphisms += len(matches)
                        print(f"子图 {sg} 中模板 {pattern_idx} 找到 {len(matches)} 个匹配")
                        
                        for match in matches:
                            # 映射回原始图中的顶点
                            mapped_match = [graph_vertex_mapping_reversed[(sg, v)] for v in match]
                            
                            # 收集匹配的详细信息
                            match_details = {
                                "vertices": mapped_match,
                                "vertex_labels": [subgraph.vs[v]["label"] for v in match],
                                "edge_details": []
                            }
                            
                            # 收集边的信息
                            for i, v1 in enumerate(match):
                                for j, v2 in enumerate(match):
                                    if i < j:
                                        edges = subgraph.get_eids(v1, v2)
                                        for eid in edges:
                                            edge = subgraph.es[eid]
                                            match_details["edge_details"].append({
                                                "source": mapped_match[i],
                                                "target": mapped_match[j],
                                                "label": edge["label"]
                                            })
                            
                            pattern_matches["matches"].append(match_details)
                        
                        subgraph_results["pattern_matches"].append(pattern_matches)
                    
                    detailed_subisomorphisms.append(subgraph_results)
                
                metadata = {
                    "total_counts": total_subisomorphisms,
                    "subgraph_results": detailed_subisomorphisms
                }

                    # 在生成图后添加详细输出
                print("\n===== 最终图结构 =====")
                print(f"顶点: {[v['label'] for v in graph.vs]}")
                print("边:")
                for edge in graph.es:
                    print(f"  {edge.source} -> {edge.target}: 标签={edge['label']}")
            else:
                counts = 0
                for sg, subgraph in enumerate(subgraphs):
                    for pattern_idx, pattern in enumerate(self.patterns):
                        counts += self.pattern_checkers[pattern_idx].count_subisomorphisms(subgraph, pattern)
                metadata = {"counts": counts, "subisomorphisms": list()}
            
            if max_pattern_counts != -1 and metadata["total_counts"] > max_pattern_counts:
                alpha = alpha * max_pattern_counts / metadata["total_counts"] * DECAY
                return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
                    alpha=alpha, max_subgraph=max_subgraph, max_pattern_counts=max_pattern_counts,
                    return_subisomorphisms=return_subisomorphisms)
            # # 如果模式计数超过限制，重新生成
            # if metadata["total_counts"] > max_pattern_counts:
            #     alpha = alpha * max_pattern_counts / metadata["total_counts"] * DECAY
            #     return self.generate(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels,
            #         alpha=alpha, max_subgraph=max_subgraph, max_pattern_counts=max_pattern_counts,
            #         return_subisomorphisms=return_subisomorphisms)
            
            return graph, metadata
        


    def _get_pattern_edge_mapping(self, pattern):
        mapping = defaultdict(set)
        for edge in pattern.es:
            key = (edge.source, edge.target)
            mapping[key].add(edge["label"])
        return mapping
    
    def add_pattern(self, subgraph, graph_vertex_label_mapping_reversed, graph_edge_label_mapping):
        sg = subgraph["sg"]
        subisomorphism = list()
        for vertex_label in self.pattern.vs["label"]:
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
        return new_ecount

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
            edge_label = np.random.randint(0, graph_number_of_edge_labels)
            
            if get_direction():
                x = (subgraph1.vs[v1]["label"], subgraph2.vs[v2]["label"])
                y = (sg1, v1, sg2, v2)
            else:
                x = (subgraph2.vs[v2]["label"], subgraph1.vs[v1]["label"])
                y = (sg2, v2, sg1, v1)
                
            # 检查边标签是否在模板中（使用 pattern_meta）
            edge_in_pattern = False
            for meta in self.pattern_meta:
                if edge_label in meta["vertex_edge_label_mapping"].get(x, set()):
                    edge_in_pattern = True
                    break
            
            if edge_in_pattern:
                invalid_cnt += 1
                continue
                
            graph_edge_labels = graph_edge_label_mapping.get(y, set())
            if edge_label in graph_edge_labels:
                invalid_cnt += 1
                continue
            
            # 添加新边
            graph_edge_labels.add(edge_label)
            graph_edge_label_mapping[y] = graph_edge_labels
            
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
