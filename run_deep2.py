import numpy as np
import argparse
import os
import igraph as ig
import json
from multiprocessing import Pool
from utils import generate_labels, get_direction
from pattern_checker import PatternChecker
from graph_generator_deep2 import GraphGenerator
from pattern_generator import generate_patterns
from time import sleep
from tqdm import tqdm
import hashlib
from collections import Counter


 
import sys
import logging
 
class PrintToLog(logging.StreamHandler):
    def emit(self, record):
        log_msg = self.format(record)
        sys.__stdout__.write(log_msg + '\n')  # 将日志消息写回标准输出（可选）
        self.flush()
 


DEBUG_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.3],

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

    "save_data_dir": r"E:\wyh\layout\code\NeuralSubgraphCounting\data\deep",
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

def generate_graphs(graph_generator, number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
    alpha, max_pattern_counts, max_subgraph, number_of_graphs, save_graph_dir, save_metadata_dir):
    graphs_id = "G_N%d_E%d_NL%d_EL%d_A%.2f" % (
        number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels, alpha)
    # print(graphs_id)
    for g in range(number_of_graphs):
        graph, metadata = graph_generator.generate( 
            number_of_graph_vertices, number_of_graph_edges, number_of_graph_vertex_labels, number_of_graph_edge_labels,
            alpha, max_pattern_counts=max_pattern_counts, max_subgraph=max_subgraph, return_subisomorphisms=True)
        graph.write(os.path.join(save_graph_dir, graphs_id + "_%d.gml" % (g)))
        with open(os.path.join(save_metadata_dir, graphs_id + "_%d.meta" % (g)), "w") as f:
            json.dump(metadata, f)
    return graphs_id

if __name__ == "__main__":


    # # 配置日志
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     filename='app.log',  # 日志文件名
    #     filemode='a'         # 追加模式
    # )
    
    # # 创建自定义处理器并添加到root logger
    # custom_handler = PrintToLog(sys.stdout)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # custom_handler.setFormatter(formatter)
    # logging.getLogger().addHandler(custom_handler)
    
    # # 示例：所有print会被捕获并记录到日志文件
    # print("This is a test message.")

    parser = argparse.ArgumentParser(description='Pattern generation')
    parser.add_argument('--config-file', type=str, default='./generator/config.json')
    args = parser.parse_args()
    CONFIG = json.load(open(args.config_file))

   

    # 创建保存目录
    save_pattern_dir = os.path.join(CONFIG["save_data_dir"], "patterns")
    save_graph_dir = os.path.join(CONFIG["save_data_dir"], "graphs")
    save_metadata_dir = os.path.join(CONFIG["save_data_dir"], "metadata")
    print(f"Saving data to: {CONFIG['save_data_dir']}, save_pattern_dir: {save_pattern_dir}, save_graph_dir: {save_graph_dir}, save_metadata_dir: {save_metadata_dir}")
    os.makedirs(CONFIG["save_data_dir"], exist_ok=True)
    os.makedirs(save_pattern_dir, exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)

    np.random.seed(0)

    # === 生成所有可能的模板并保存到文件 ===
    pattern_files = {}  # 用于存储所有模板文件的字典
    patterns_counter = {}
    for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
        for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
            if number_of_pattern_vertex_labels > number_of_pattern_vertices:
                continue
            for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
                if number_of_pattern_edges < number_of_pattern_vertices - 1:
                    continue
                if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices:
                    continue
                for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
                    if number_of_pattern_edge_labels > number_of_pattern_edges:
                        continue
                    patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
                        number_of_pattern_vertices, number_of_pattern_edges, 
                        number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
                    
                    # 为每种参数组合生成多个模板
                    generated_files = []
                    for p_idx in range(CONFIG["number_of_patterns"]):
                        pattern_file = os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % p_idx)
                        
                        # 如果模板文件不存在，则生成
                        if not os.path.exists(pattern_file):
                            patterns = generate_patterns(
                                number_of_pattern_vertices, number_of_pattern_edges,
                                number_of_pattern_vertex_labels, number_of_pattern_edge_labels,
                                1)  # 每次只生成一个模板
                            
                            # 保存模板
                            patterns[0].write(pattern_file)
                        
                        generated_files.append(pattern_file)
                    
                    pattern_files[patterns_id] = generated_files
                    patterns_counter[patterns_id] = len(generated_files)
                    print(f"Generated {len(generated_files)} patterns for ID: {patterns_id}")

    print(f"Total pattern types: {len(pattern_files)}")

    graph_cnt = 0
    pool = Pool(CONFIG["num_workers"])
    results = list()

    # 获取所有可能的pattern_id
    all_pattern_ids = list(pattern_files.keys())
    for _ in tqdm(range(CONFIG["number_of_graphs"]), desc="Generating graphs"):
        # 随机选择1-10个模板
        num_patterns = np.random.randint(1, 11)
        selected_patterns = []
        selected_ids = []
        
        # 随机选择模板
        for _ in range(num_patterns):


            n_vertices = np.random.choice(CONFIG["number_of_pattern_vertices"])
            n_edges = np.random.choice(CONFIG["number_of_pattern_edges"])
            n_vlabels = np.random.choice(CONFIG["number_of_pattern_vertex_labels"])
            n_elabels = np.random.choice(CONFIG["number_of_pattern_edge_labels"])
            # 随机选择一个pattern_id
            pattern_id = np.random.choice(all_pattern_ids)
            
            # 从该pattern_id对应的文件中随机选择一个模板
            # 生成或获取现有模板
            
            pattern_file = os.path.join(save_pattern_dir, pattern_id + f"_{np.random.randint(CONFIG['number_of_patterns'])}.gml")
            print(f"Selected pattern file: {pattern_file}")
            pattern = ig.read(pattern_file)
            pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
            pattern.es["label"] = [int(x) for x in pattern.es["label"]]
            selected_patterns.append(pattern)
            
            
            selected_ids.append(pattern_id)
            # 设置图参数 - 确保足够大以容纳模板
        
        
        # 为选择的模板生成唯一标识
        pattern_hash = hashlib.md5("+".join(selected_ids).encode()).hexdigest()[:8]

        min_graph_vertices = max([p.vcount() for p in selected_patterns]) 
        min_graph_edges = max([p.ecount() for p in selected_patterns]) 
        
        n_graph_vertices = max(
            np.random.choice(CONFIG["number_of_graph_vertices"]),
            min_graph_vertices * 2
        )
        
        n_graph_edges = max(
            np.random.choice(CONFIG["number_of_graph_edges"]),
            min_graph_edges * 3
        )
        
        # 确保标签数足够
        required_vertex_labels = max(len(Counter(p.vs["label"])) for p in selected_patterns)
        required_edge_labels = max(len(Counter(p.es["label"])) for p in selected_patterns)
        
        n_graph_vlabels = max(
            np.random.choice(CONFIG["number_of_graph_vertex_labels"]),
            required_vertex_labels
        )
        
        n_graph_elabels = max(
            np.random.choice(CONFIG["number_of_graph_edge_labels"]),
            required_edge_labels
        )
        
        alpha_val = np.random.choice(CONFIG["alphas"])
        
        # 创建支持多个模板的graph_generator
        graph_generator = GraphGenerator(selected_patterns)
        # 为选择的模板生成唯一标识
        pattern_hash = hashlib.md5("+".join(selected_ids).encode()).hexdigest()[:8]
        
        # 随机选择图参数
        # n_graph_vertices = np.random.choice(CONFIG["number_of_graph_vertices"])
        # n_graph_edges = np.random.choice(CONFIG["number_of_graph_edges"])
        # n_graph_vlabels = np.random.choice(CONFIG["number_of_graph_vertex_labels"])
        # n_graph_elabels = np.random.choice(CONFIG["number_of_graph_edge_labels"])
        # alpha_val = np.random.choice(CONFIG["alphas"])
        
        # 创建图生成器（支持多个模板）
        # graph_generator = GraphGenerator(selected_patterns)
        
        # 为模板组创建唯一保存目录
        save_graph_dir_p = os.path.join(save_graph_dir, f"num_patterns_{num_patterns}_{pattern_hash}")
        save_metadata_dir_p = os.path.join(save_metadata_dir, f"num_patterns_{num_patterns}_{pattern_hash}")
        os.makedirs(save_graph_dir_p, exist_ok=True)
        os.makedirs(save_metadata_dir_p, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"开始生成图 {_+1}/{CONFIG['number_of_graphs']}")
        print(f"使用 {len(selected_patterns)} 个模板:")
        for i, p in enumerate(selected_patterns):
            print(f"  模板 {i}: {p.summary()}")
            print(f"    顶点标签: {p.vs['label']}")
            print(f"    边: {[(e.source, e.target, e['label']) for e in p.es]}")

        print(f"图参数: 顶点数={n_graph_vertices}, 边数={n_graph_edges}, "
          f"顶点标签数={n_graph_vlabels}, 边标签数={n_graph_elabels}")
        # 提交生成任务
        results.append(
            pool.apply_async(generate_graphs, args=(
                graph_generator, n_graph_vertices, n_graph_edges,
                n_graph_vlabels, n_graph_elabels,
                alpha_val, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
                CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
        
        graph_cnt += CONFIG["number_of_graphs"]
    # graph_cnt = 0
    # pool = Pool(CONFIG["num_workers"])
    # results = list()
    # for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
    #     for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
    #         if number_of_pattern_vertex_labels > number_of_pattern_vertices:
    #             continue
    #         for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
    #             if number_of_pattern_edges < number_of_pattern_vertices - 1: # not connected
    #                 continue
    #             if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices: # too dense
    #                 continue
    #             for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
    #                 if number_of_pattern_edge_labels > number_of_pattern_edges:
    #                     continue
    #                 patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
    #                     number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
    #                 graph_generators = list()
    #                 for p in range(CONFIG["number_of_patterns"]):
    #                     pattern = ig.read(os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % (p)))
    #                     pattern.vs["label"] = [int(x) for x in pattern.vs["label"]]
    #                     pattern.es["label"] = [int(x) for x in pattern.es["label"]]
    #                     pattern.es["key"] = [int(x) for x in pattern.es["key"]]
    #                     graph_generators.append(GraphGenerator(pattern))
    #                 for alpha in CONFIG["alphas"]:
    #                     for number_of_graph_vertices in CONFIG["number_of_graph_vertices"]:
    #                         if number_of_graph_vertices < number_of_pattern_vertices:
    #                             continue
    #                         for number_of_graph_vertex_labels in CONFIG["number_of_graph_vertex_labels"]:
    #                             if number_of_graph_vertex_labels > number_of_graph_vertices:
    #                                 continue
    #                             if number_of_graph_vertex_labels < number_of_pattern_vertex_labels:
    #                                 continue
    #                             for number_of_graph_edges in CONFIG["number_of_graph_edges"]:
    #                                 if number_of_graph_edges < number_of_graph_vertices - 1: # not connected
    #                                     continue
    #                                 if number_of_graph_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_graph_vertices: # too dense
    #                                     continue
    #                                 if number_of_graph_edges < number_of_pattern_edges:
    #                                     continue
    #                                 for number_of_graph_edge_labels in CONFIG["number_of_graph_edge_labels"]:
    #                                     if number_of_graph_edge_labels > number_of_graph_edges:
    #                                         continue
    #                                     if number_of_graph_edge_labels < number_of_pattern_edge_labels:
    #                                         continue
    #                                     for p, graph_generator in enumerate(graph_generators):
    #                                         save_graph_dir_p = os.path.join(save_graph_dir, patterns_id + "_%d" % (p))
    #                                         save_metadata_dir_p = os.path.join(save_metadata_dir, patterns_id + "_%d" % (p))
    #                                         if not os.path.isdir(save_graph_dir_p):
    #                                             os.mkdir(save_graph_dir_p)
    #                                         if not os.path.isdir(save_metadata_dir_p):
    #                                             os.mkdir(save_metadata_dir_p)
    #                                         results.append(
    #                                             pool.apply_async(generate_graphs, args=(
    #                                                 graph_generator, number_of_graph_vertices, number_of_graph_edges,
    #                                                 number_of_graph_vertex_labels, number_of_graph_edge_labels,
    #                                                 alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
    #                                                 CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
    #                                         # generate_graphs(
    #                                         #         graph_generator, number_of_graph_vertices, number_of_graph_edges,
    #                                         #         number_of_graph_vertex_labels, number_of_graph_edge_labels,
    #                                         #         alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
    #                                         #         CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)
    #                                         graph_cnt += CONFIG["number_of_graphs"]
    pool.close()
    # pool.join()
    for x in tqdm(results):
        x.get()
    print("%d graphs generation finished!" % (graph_cnt))
