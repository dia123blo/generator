import numpy as np
import argparse
import os
import igraph as ig
import json
from multiprocessing import Pool
from utils import generate_labels, get_direction
from pattern_checker_P import PatternChecker
from graph_generator_Patterns_graph_gen import GraphGenerator
from pattern_generator_P import generate_patterns
from time import sleep
from tqdm import tqdm


DEBUG_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.5],

    "number_of_patterns": 1,
    "number_of_pattern_vertices": [3, 4],
    "number_of_pattern_edges": [2, 4],
    "number_of_pattern_vertex_labels": [2, 4],
    "number_of_pattern_edge_labels": [2, 4],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [16, 64],
    "number_of_graph_edges": [16, 64, 256],
    "number_of_graph_vertex_labels": [4, 8],
    "number_of_graph_edge_labels": [4, 8],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 1024,

    "save_data_dir": r"./data/debug_Patterns_graph_gen",
    "num_workers": 16
}

SMALL_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    "number_of_patterns": 3,
    "number_of_pattern_vertices": [3, 4, 8],
    "number_of_pattern_edges": [2, 4, 8],
    "number_of_pattern_vertex_labels": [2, 4, 8],
    "number_of_pattern_edge_labels": [2, 4, 8],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [8, 16, 32, 64],
    "number_of_graph_edges": [8, 16, 32, 64, 128, 256],
    "number_of_graph_vertex_labels": [4, 8, 16],
    "number_of_graph_edge_labels": [4, 8, 16],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 1024,

    "save_data_dir": r"/data/xliucr/SubIsoCnt/small",
    "num_workers": 16
}

LARGE_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.05, 0.1, 0.15],
    
    "number_of_patterns": 2,
    "number_of_pattern_vertices": [3, 4, 8, 16],
    "number_of_pattern_edges": [2, 4, 8, 16],
    "number_of_pattern_vertex_labels": [2, 4, 8, 16],
    "number_of_pattern_edge_labels": [2, 4, 8, 16],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [64, 128, 256, 512],
    "number_of_graph_edges": [64, 128, 256, 512, 1024, 2048],
    "number_of_graph_vertex_labels": [16, 32, 64],
    "number_of_graph_edge_labels": [16, 32, 64],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 4096,

    "save_data_dir": r"/data/xliucr/SubIsoCnt/large",
    "num_workers": 16
}

CONFIG = DEBUG_CONFIG

# def generate_graphs(graph_generator, number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
#     alpha, max_pattern_counts, max_subgraph, number_of_graphs, save_graph_dir, save_metadata_dir):
#     graphs_id = "G_N%d_E%d_NL%d_EL%d_A%.2f" % (
#         number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels, alpha)
#     # print(graphs_id)
#     for g in range(number_of_graphs):
#         graph, metadata = graph_generator.generate( 
#             number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
#             alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph, return_subisomorphisms=True)
#         graph.write(os.path.join(save_graph_dir, graphs_id + "_%d.gml" % (g)))
#         with open(os.path.join(save_metadata_dir, graphs_id + "_%d.meta" % (g)), "w") as f:
#             json.dump(metadata, f)
#     return graphs_id
def generate_graphs(graph_generator, number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
    alpha, max_pattern_counts, max_subgraph, number_of_graphs, save_graph_dir, save_metadata_dir):
    
    graphs_id = "G_N%d_E%d_NL%d_EL%d_A%.2f" % (
        number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels, alpha)
    
    for g in range(number_of_graphs):
        graph, metadata = graph_generator.generate( 
            number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
            alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph, return_subisomorphisms=True)
        
        # 确保图属性是GML兼容类型
        graph.vs["label"] = [str(x) for x in graph.vs["label"]]
        graph.es["label"] = [str(x) for x in graph.es["label"]]
        if "key" in graph.es.attributes():
            graph.es["key"] = [str(x) for x in graph.es["key"]]
        
        graph.write(os.path.join(save_graph_dir, graphs_id + "_%d.gml" % (g)))
        with open(os.path.join(save_metadata_dir, graphs_id + "_%d.meta" % (g)), "w") as f:
            json.dump(metadata, f)
    return graphs_id

if __name__ == "__main__":
    # 使用你实际的pattern路径
    save_pattern_dir = r"E:\wyh\layout\code\output\pattern"
    save_graph_dir = os.path.join(CONFIG["save_data_dir"], "graphs")
    save_metadata_dir = os.path.join(CONFIG["save_data_dir"], "metadata")
    os.makedirs(CONFIG["save_data_dir"], exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)

    np.random.seed(0)

    # 获取所有已有的pattern文件
    pattern_files = [f for f in os.listdir(save_pattern_dir) if f.endswith('.gml')]
    pattern_names = []
    for f in pattern_files:
        # 提取pattern名称，例如从"Current_Mirror_000000.gml"提取"Current_Mirror"
        name_parts = f.split('_')
        if len(name_parts) >= 2:
            # 处理像"Voltage_Reference_I"这样的名称
            pattern_name = '_'.join(name_parts[:-1])
        else:
            # 处理像"Differential_Pair.gml"这样的名称
            pattern_name = '.'.join(f.split('.')[:-1])
        
        if pattern_name not in pattern_names:
            pattern_names.append(pattern_name)
    
    print("Found pattern names:", pattern_names)

    graph_cnt = 0
    pool = Pool(CONFIG["num_workers"])
    results = list()
    
    # 针对每个pattern名称生成graphs
    for pattern_name in pattern_names:
        graph_generators = list()
        
        # 查找该pattern的所有变体（例如Current_Mirror_000000.gml, Current_Mirror_000001.gml等）
        pattern_variants = [f for f in pattern_files if f.startswith(pattern_name)]
        
        for p, variant in enumerate(pattern_variants):
            pattern_path = os.path.join(save_pattern_dir, variant)
            try:
                # 处理BOM问题的函数
                def read_gml_without_bom(file_path):
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        # 移除BOM标记 (UTF-8 BOM: EF BB BF)
                        if content.startswith(b'\xef\xbb\xbf'):
                            content = content[3:]
                        # 创建临时文件进行读取
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w+', suffix='.gml', delete=False) as tmp_f:
                            tmp_f.write(content.decode('utf-8'))
                            tmp_path = tmp_f.name
                    
                    # 读取临时文件
                    graph = ig.read(tmp_path)
                    # 清理临时文件
                    os.unlink(tmp_path)
                    return graph
                
                # 尝试读取pattern文件
                pattern = read_gml_without_bom(pattern_path)
                
                # 正确处理label，保持为字符串形式
                # pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
                # pattern.es["label"] = [int(x) for x in pattern.es["label"]]
                
                # 保持label为字符串，但确保它们是有效的
                pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
                if pattern.ecount() > 0:  # 只有当图有边时才处理边标签
                    pattern.es["label"] = [str(x) for x in pattern.es["label"]]
                
                graph_generators.append(GraphGenerator(pattern))
                print(f"Loaded pattern: {variant}")
            except Exception as e:
                print(f"Error loading pattern {variant}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not graph_generators:
            print(f"No valid patterns found for {pattern_name}, skipping...")
            continue
            
        # 为每个pattern变体生成graphs
        for p, graph_generator in enumerate(graph_generators):
            # 创建pattern特定的目录名
            patterns_id = pattern_name
            save_graph_dir_p = os.path.join(save_graph_dir, patterns_id + "_%d" % (p))
            save_metadata_dir_p = os.path.join(save_metadata_dir, patterns_id + "_%d" % (p))
            
            if not os.path.isdir(save_graph_dir_p):
                os.makedirs(save_graph_dir_p, exist_ok=True)
            if not os.path.isdir(save_metadata_dir_p):
                os.makedirs(save_metadata_dir_p, exist_ok=True)
                
            # 生成graphs
            for alpha in CONFIG["alphas"]:
                for number_of_graph_vertices in CONFIG["number_of_graph_vertices"]:
                    for number_of_graph_vertex_labels in CONFIG["number_of_graph_vertex_labels"]:
                        if number_of_graph_vertex_labels > number_of_graph_vertices:
                            continue
                        for number_of_graph_edges in CONFIG["number_of_graph_edges"]:
                            if number_of_graph_edges < number_of_graph_vertices - 1: # not connected
                                continue
                            if number_of_graph_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_graph_vertices: # too dense
                                continue
                            for number_of_graph_edge_labels in CONFIG["number_of_graph_edge_labels"]:
                                if number_of_graph_edge_labels > number_of_graph_edges:
                                    continue
                                results.append(
                                    pool.apply_async(generate_graphs, args=(
                                        graph_generator, number_of_graph_vertices, number_of_graph_edges,
                                        number_of_graph_vertex_labels, number_of_graph_edge_labels,
                                        alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
                                        CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
                                graph_cnt += CONFIG["number_of_graphs"]
                                
    pool.close()
    for x in tqdm(results):
        x.get()
    print("%d graphs generation finished!" % (graph_cnt))