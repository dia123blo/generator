# import networkx as nx
import igraph as ig
import argparse
import numpy as np
import os
from utils_P import generate_edge_label_by_device_types,generate_labels, generate_tree, get_direction, str2bool,generate_vf_random_in_range_list,generate_ef_random_in_range_list
from collections import Counter, defaultdict
from time import time

def generate_patterns(number_of_vertices, number_of_edges, number_of_vertex_labels, number_of_edge_labels, number_of_patterns):
    patterns = []
    
    for p in range(number_of_patterns):
        pattern = ig.Graph(directed=True)
        
        # 生成节点标签（器件编码）
        vertex_labels = generate_vf_random_in_range_list(number_of_vertices)
        
        # 生成树结构
        pattern = generate_tree(number_of_vertices, directed=True)
        pattern.vs["label"] = vertex_labels
        
        edge_label_mapping = defaultdict(set)
        ecount = pattern.ecount()
        new_edges = list()
        
        # 为现有边生成符合规则的边标签
        for e, (u, v) in enumerate(pattern.get_edgelist()):
            source_label = pattern.vs[u]["label"]
            target_label = pattern.vs[v]["label"]
            edge_label = generate_edge_label_by_device_types(source_label, target_label)
            edge_label_mapping[(u, v)].add(edge_label)
        
        # 添加额外边
        while ecount < number_of_edges:
            u = np.random.randint(0, number_of_vertices)
            v = np.random.randint(0, number_of_vertices)
            
            source_label = pattern.vs[u]["label"]
            target_label = pattern.vs[v]["label"]
            edge_label = generate_edge_label_by_device_types(source_label, target_label)
            
            if edge_label in edge_label_mapping[(u, v)]:
                continue
                
            new_edges.append((u, v))
            edge_label_mapping[(u, v)].add(edge_label)
            ecount += 1
            
        pattern.add_edges(new_edges)
        
        # 设置边标签
        # 设置边标签
        edge_labels = []
        for u, v in pattern.get_edgelist():
            edge_labels.append(next(iter(edge_label_mapping[(u, v)])))
        pattern.es["label"] = edge_labels
        
        # 确保属性是GML兼容类型
        pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
        pattern.es["label"] = [str(x) for x in pattern.es["label"]]
        
        patterns.append(pattern)
    
    return patterns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_of_vertices", type=int, default=3)
    parser.add_argument("--number_of_edges", type=int, default=3)
    parser.add_argument("--number_of_vertex_labels", type=int, default=2)
    parser.add_argument("--number_of_edge_labels", type=int, default=2)
    parser.add_argument("--number_of_patterns", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="patterns")
    parser.add_argument("--save_png", type=str2bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    patterns = generate_patterns(args.number_of_vertices, args.number_of_edges,
        args.number_of_vertex_labels, args.number_of_edge_labels,
        args.number_of_patterns)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for p, pattern in enumerate(patterns):
            pattern_id = "P_N%d_E%d_NL%d_EL%d_%d" % (
                args.number_of_vertices, args.number_of_edges, args.number_of_vertex_labels, args.number_of_edge_labels, p)
            filename = os.path.join(args.save_dir, pattern_id)
            # nx.nx_pydot.write_dot(pattern, filename + ".dot")
            pattern.write(filename + ".gml")
            if args.save_png:
                ig.plot(pattern, filename + ".png")

