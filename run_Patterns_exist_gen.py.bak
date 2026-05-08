import numpy as np
import argparse
import os
import igraph as ig
import json
from multiprocessing import Pool
from utils import generate_labels, get_direction
from pattern_checker_P import PatternChecker
from graph_generator_Patterns_exist_gen import GraphGenerator
from pattern_generator_P import generate_patterns
from time import sleep
from tqdm import tqdm


DEBUG_CONFIG = {
    "max_subgraph": 512,

    "alphas": [0.5],

    "number_of_patterns": 1,
    "number_of_pattern_vertices": [3,4,5],
    "number_of_pattern_edges": [2, 4, 8, 14],
    "number_of_pattern_vertex_labels": [2, 4],
    "number_of_pattern_edge_labels": [2, 4],

    "number_of_graphs": 10, # train:dev:test = 8:1:1
    "number_of_graph_vertices": [16, 64],
    "number_of_graph_edges": [16, 64, 256],
    "number_of_graph_vertex_labels": [4, 8],
    "number_of_graph_edge_labels": [4, 8, 16],

    "max_ratio_of_edges_vertices": 4,
    "max_pattern_counts": 1024,

    "save_data_dir": r"./data/debug_Patterns_exist_gen",
    "num_workers": 16,
    "existing_pattern_dir": r"E:\wyh\layout\code\output\pattern"  # 新增配置项：已有pattern的目录路径
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
    "num_workers": 16,
    "existing_pattern_dir": None  # 新增配置项：已有pattern的目录路径
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
    "num_workers": 16,
    "existing_pattern_dir": None  # 新增配置项：已有pattern的目录路径
}

CONFIG = DEBUG_CONFIG

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
        graph.es["label"] = [str(x) for x in graph.es["label"]]  # Convert labels to strings for GML compatibility
        
        # 将生成的图和元数据添加到列表中（关键步骤）
        graphs.append(graph)
        metadatas.append(metadata)
        
        # 验证metadata的准确性
        subisomorphisms = metadata.get("subisomorphisms", [])
        counts = metadata.get("counts", 0)
        
        # 双重验证：使用pattern_checker重新计数
        pattern_checker = PatternChecker()
        actual_count = pattern_checker.count_subisomorphisms(graph, graph_generator.pattern)
        
        if actual_count != counts:
            print(f"警告: 图 {graphs_id}_{g} 的pattern计数不匹配: "
                  f"metadata={counts}, 实际={actual_count}")
            # 更新metadata
            metadata["counts"] = actual_count
            metadata["verified_count"] = actual_count
            # 更新列表中的metadata
            metadatas[g] = metadata
            
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

def parse_gml_content(content):
    """直接解析GML内容"""
    try:
        # 使用igraph的Graph.Read_GML方法
        import io
        pattern = ig.Graph.Read_GML(io.StringIO(content))
        return pattern
    except Exception as e:
        print(f"Error parsing GML content: {e}")
        return None

def load_existing_patterns(pattern_dir):
    """从指定目录加载已有的pattern文件"""
    patterns = []
    pattern_files = [f for f in os.listdir(pattern_dir) if f.endswith('.gml')]
    
    for pattern_file in pattern_files:
        try:
            pattern_path = os.path.join(pattern_dir, pattern_file)
            print(f"Attempting to load pattern: {pattern_file}")
            
            # 首先检查文件是否为空或损坏
            if not os.path.exists(pattern_path) or os.path.getsize(pattern_path) == 0:
                print(f"Warning: Pattern file {pattern_file} is empty or does not exist")
                continue
                
            # 尝试读取文件内容
            with open(pattern_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                print(f"File content preview (first 500 chars): {content[:500]}")
                
                # 清理文件内容
                import re
                # 移除BOM头
                if content.startswith('\ufeff'):
                    content = content[1:]
                
                # 移除开头的任何非字母数字字符
                content = re.sub(r'^[^a-zA-Z]*', '', content)
            
            # 将清理后的内容写入临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gml', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # 尝试从临时文件加载pattern
                pattern = ig.read(temp_path)
                
                # 检查pattern是否有效
                if pattern.vcount() == 0:
                    print(f"Warning: Pattern {pattern_file} has no vertices")
                    continue
                    
                # 确保属性是字符型
                pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
                if pattern.ecount() > 0:
                    pattern.es["label"] = [str(x) for x in pattern.es["label"]]
                
                # 提取pattern参数信息
                pattern_info = {
                    'pattern': pattern,
                    'file': pattern_file,
                    'path': pattern_path,
                    'number_of_pattern_vertices': pattern.vcount(),
                    'number_of_pattern_edges': pattern.ecount(),
                    'number_of_pattern_vertex_labels': len(set(pattern.vs["label"])),
                    'number_of_pattern_edge_labels': len(set(pattern.es["label"])) if pattern.ecount() > 0 else 0
                }
                
                patterns.append(pattern_info)
                print(f"Successfully loaded pattern: {pattern_file} (vcount: {pattern.vcount()}, ecount: {pattern.ecount()})")
                
            finally:
                # 删除临时文件
                os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error loading pattern {pattern_file}: {e}")
            # 尝试使用备用方法创建简单的pattern
            simple_pattern = create_simple_pattern_from_filename(pattern_file)
            if simple_pattern:
                patterns.append(simple_pattern)
                print(f"Created simple pattern from filename: {pattern_file}")
    
    return patterns
def create_simple_pattern_from_filename(filename):
    """根据文件名创建简单的pattern"""
    try:
        # 从文件名提取参数
        filename_parts = os.path.splitext(filename)[0].split('_')
        
        # 默认值
        num_vertices = 3
        num_edges = 2
        num_vertex_labels = 2
        num_edge_labels = 2
        
        # 从文件名解析参数
        for part in filename_parts:
            if part.startswith('N') and len(part) > 1 and part[1:].isdigit():
                num_vertices = int(part[1:])
            elif part.startswith('E') and len(part) > 1 and part[1:].isdigit():
                num_edges = int(part[1:])
            elif part.startswith('NL') and len(part) > 2 and part[2:].isdigit():
                num_vertex_labels = int(part[2:])
            elif part.startswith('EL') and len(part) > 2 and part[2:].isdigit():
                num_edge_labels = int(part[2:])
        
        # 创建简单的pattern
        pattern = ig.Graph(directed=True)
        
        # 添加顶点
        pattern.add_vertices(num_vertices)
        
        # 生成顶点标签（字符型）
        vertex_labels = [f"v{i}" for i in range(num_vertices)]
        pattern.vs["label"] = vertex_labels
        
        # 添加边（如果可能）
        if num_edges > 0 and num_vertices > 1:
            # 创建简单路径
            edges = []
            edge_labels = []
            
            # 确保不超过最大可能的边数
            max_edges = min(num_edges, num_vertices * (num_vertices - 1))
            
            for i in range(max_edges):
                source = i % num_vertices
                target = (i + 1) % num_vertices
                if source != target and (source, target) not in edges:
                    edges.append((source, target))
                    edge_labels.append(f"e{i}")
            
            if edges:
                pattern.add_edges(edges)
                pattern.es["label"] = edge_labels
        
        pattern_info = {
            'pattern': pattern,
            'file': filename,
            'path': f"generated_from_{filename}",
            'number_of_pattern_vertices': pattern.vcount(),
            'number_of_pattern_edges': pattern.ecount(),
            'number_of_pattern_vertex_labels': len(set(pattern.vs["label"])),
            'number_of_pattern_edge_labels': len(set(pattern.es["label"])) if pattern.ecount() > 0 else 0
        }
        
        return pattern_info
        
    except Exception as e:
        print(f"Error creating simple pattern from {filename}: {e}")
        return None

# if __name__ == "__main__":
#     save_pattern_dir = os.path.join(CONFIG["save_data_dir"], "patterns")
#     save_graph_dir = os.path.join(CONFIG["save_data_dir"], "graphs")
#     save_metadata_dir = os.path.join(CONFIG["save_data_dir"], "metadata")
#     os.makedirs(CONFIG["save_data_dir"], exist_ok=True)
#     os.makedirs(save_pattern_dir, exist_ok=True)
#     os.makedirs(save_graph_dir, exist_ok=True)
#     os.makedirs(save_metadata_dir, exist_ok=True)

#     np.random.seed(0)

#     # 检查是否使用已有pattern
#     if CONFIG.get("existing_pattern_dir") and os.path.exists(CONFIG["existing_pattern_dir"]):
#         print("Using existing patterns from:", CONFIG["existing_pattern_dir"])
#         existing_patterns = load_existing_patterns(CONFIG["existing_pattern_dir"])
        
#         # 如果已有pattern目录不是当前保存目录，复制pattern文件到当前目录
#         if CONFIG["existing_pattern_dir"] != save_pattern_dir:
#             for pattern_info in existing_patterns:
#                 dest_path = os.path.join(save_pattern_dir, pattern_info['file'])
#                 if not os.path.exists(dest_path):
#                     import shutil
#                     shutil.copy2(pattern_info['path'], dest_path)
#                     print(f"Copied pattern to: {dest_path}")
#     else:
#         # 原有pattern生成逻辑
#         print("Generating new patterns...")
#         pattern_cnt = 0
#         for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
#             for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
#                 if number_of_pattern_vertex_labels > number_of_pattern_vertices:
#                     continue
#                 for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
#                     if number_of_pattern_edges < number_of_pattern_vertices - 1: # not connected
#                         continue
#                     if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices: # too dense
#                         continue
#                     for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
#                         if number_of_pattern_edge_labels > number_of_pattern_edges:
#                             continue
#                         patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
#                             number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
#                         for p, pattern in enumerate(generate_patterns(
#                             number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels,
#                             CONFIG["number_of_patterns"])):
#                             # 确保模式图属性是字符型
#                             pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
#                             pattern.es["label"] = [str(x) for x in pattern.es["label"]]  # 边标签改为字符型
#                             if "key" in pattern.es.attributes():
#                                 pattern.es["key"] = [str(x) for x in pattern.es["key"]]
#                             pattern.write(os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % (p)))
#                         pattern_cnt += CONFIG["number_of_patterns"]
#                         print("patterns_id", patterns_id)
#         print("%d patterns generation finished!" % (pattern_cnt))
        
#         # 加载新生成的patterns
#         existing_patterns = load_existing_patterns(save_pattern_dir)

#         # 根据已有patterns生成graphs
#     graph_cnt = 0
#     pool = Pool(CONFIG["num_workers"])
#     results = list()
    
#     # 为每个pattern文件单独处理，而不是按组处理
#     for p, pattern_info in enumerate(existing_patterns):
#         pattern = pattern_info['pattern']
#         pattern_file = pattern_info['file']
#         # 使用完整的文件名（不含扩展名）作为基础名，保持原始pattern名称
#         base_name = os.path.splitext(pattern_file)[0]
        
#         # 确保pattern的边标签是字符型
#         if pattern.ecount() > 0:
#             pattern.es["label"] = [str(x) for x in pattern.es["label"]]
        
#         graph_generator = GraphGenerator(pattern)
        
#         # 使用pattern的实际参数而不是配置中的参数
#         pattern_vertices = pattern.vcount()
#         pattern_edges = pattern.ecount()
#         pattern_vertex_labels = len(set(pattern.vs["label"]))
#         pattern_edge_labels = len(set(pattern.es["label"])) if pattern.ecount() > 0 else 0
        
#         # 为每个pattern只创建一个目录，使用原始pattern名称
#         save_graph_dir_p = os.path.join(save_graph_dir, base_name)
#         save_metadata_dir_p = os.path.join(save_metadata_dir, base_name)
#         if not os.path.isdir(save_graph_dir_p):
#             os.makedirs(save_graph_dir_p, exist_ok=True)
#         if not os.path.isdir(save_metadata_dir_p):
#             os.makedirs(save_metadata_dir_p, exist_ok=True)
        
#         for alpha in CONFIG["alphas"]:
#             for number_of_graph_vertices in CONFIG["number_of_graph_vertices"]:
#                 if number_of_graph_vertices < pattern_vertices:
#                     continue
#                 for number_of_graph_vertex_labels in CONFIG["number_of_graph_vertex_labels"]:
#                     if number_of_graph_vertex_labels > number_of_graph_vertices:
#                         continue
#                     if number_of_graph_vertex_labels < pattern_vertex_labels:
#                         continue
#                     for number_of_graph_edges in CONFIG["number_of_graph_edges"]:
#                         if number_of_graph_edges < number_of_graph_vertices - 1: # not connected
#                             continue
#                         if number_of_graph_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_graph_vertices: # too dense
#                             continue
#                         if number_of_graph_edges < pattern_edges:
#                             continue
#                         for number_of_graph_edge_labels in CONFIG["number_of_graph_edge_labels"]:
#                             if number_of_graph_edge_labels > number_of_graph_edges:
#                                 continue
#                             if number_of_graph_edge_labels < pattern_edge_labels:
#                                 continue
#                             results.append(
#                                 pool.apply_async(generate_graphs, args=(
#                                     graph_generator, number_of_graph_vertices, number_of_graph_edges,
#                                     number_of_graph_vertex_labels, number_of_graph_edge_labels,
#                                     alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
#                                     CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
#                             graph_cnt += CONFIG["number_of_graphs"]
    
#     pool.close()
#     for x in tqdm(results):
#         x.get()
#     print("%d graphs generation finished!" % (graph_cnt))
if __name__ == "__main__":
    save_pattern_dir = os.path.join(CONFIG["save_data_dir"], "patterns")
    save_graph_dir = os.path.join(CONFIG["save_data_dir"], "graphs")
    save_metadata_dir = os.path.join(CONFIG["save_data_dir"], "metadata")
    os.makedirs(CONFIG["save_data_dir"], exist_ok=True)
    os.makedirs(save_pattern_dir, exist_ok=True)
    os.makedirs(save_graph_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)

    np.random.seed(0)

    # 检查是否使用已有pattern
    if CONFIG.get("existing_pattern_dir") and os.path.exists(CONFIG["existing_pattern_dir"]):
        print("Using existing patterns from:", CONFIG["existing_pattern_dir"])
        existing_patterns = load_existing_patterns(CONFIG["existing_pattern_dir"])
        
        # 如果已有pattern目录不是当前保存目录，复制pattern文件到当前目录
        if CONFIG["existing_pattern_dir"] != save_pattern_dir:
            for pattern_info in existing_patterns:
                dest_path = os.path.join(save_pattern_dir, pattern_info['file'])
                if not os.path.exists(dest_path):
                    import shutil
                    shutil.copy2(pattern_info['path'], dest_path)
                    print(f"Copied pattern to: {dest_path}")
    else:
        # 原有pattern生成逻辑
        print("Generating new patterns...")
        pattern_cnt = 0
        for number_of_pattern_vertices in CONFIG["number_of_pattern_vertices"]:
            for number_of_pattern_vertex_labels in CONFIG["number_of_pattern_vertex_labels"]:
                if number_of_pattern_vertex_labels > number_of_pattern_vertices:
                    continue
                for number_of_pattern_edges in CONFIG["number_of_pattern_edges"]:
                    if number_of_pattern_edges < number_of_pattern_vertices - 1: # not connected
                        continue
                    if number_of_pattern_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_pattern_vertices: # too dense
                        continue
                    for number_of_pattern_edge_labels in CONFIG["number_of_pattern_edge_labels"]:
                        if number_of_pattern_edge_labels > number_of_pattern_edges:
                            continue
                        patterns_id = "P_N%d_E%d_NL%d_EL%d" % (
                            number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels)
                        for p, pattern in enumerate(generate_patterns(
                            number_of_pattern_vertices, number_of_pattern_edges, number_of_pattern_vertex_labels, number_of_pattern_edge_labels,
                            CONFIG["number_of_patterns"])):
                            # 确保模式图属性是字符型
                            pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
                            pattern.es["label"] = [str(x) for x in pattern.es["label"]]  # 边标签改为字符型
                            if "key" in pattern.es.attributes():
                                pattern.es["key"] = [str(x) for x in pattern.es["key"]]
                            pattern.write(os.path.join(save_pattern_dir, patterns_id + "_%d.gml" % (p)))
                        pattern_cnt += CONFIG["number_of_patterns"]
                        print("patterns_id", patterns_id)
        print("%d patterns generation finished!" % (pattern_cnt))
        
        # 加载新生成的patterns
        existing_patterns = load_existing_patterns(save_pattern_dir)

    # 根据已有patterns生成graphs
    graph_cnt = 0
    pool = Pool(CONFIG["num_workers"])
    results = list()
    
    # 总的子图统计信息
    total_subgraph_stats = []
    
    # 为每个pattern文件单独处理，而不是按组处理
    for p, pattern_info in enumerate(existing_patterns):
        pattern = pattern_info['pattern']
        pattern_file = pattern_info['file']
        # 使用完整的文件名（不含扩展名）作为基础名，保持原始pattern名称
        base_name = os.path.splitext(pattern_file)[0]
        
        # 确保pattern的节点和边标签都是字符串类型
        pattern.vs["label"] = [str(x) for x in pattern.vs["label"]]
        if pattern.ecount() > 0:
            pattern.es["label"] = [str(x) for x in pattern.es["label"]]
        
        graph_generator = GraphGenerator(pattern)
        
        # 使用pattern的实际参数而不是配置中的参数
        pattern_vertices = pattern.vcount()
        pattern_edges = pattern.ecount()
        pattern_vertex_labels = len(set(str(x) for x in pattern.vs["label"]))
        pattern_edge_labels = len(set(str(x) for x in pattern.es["label"])) if pattern.ecount() > 0 else 0
        
        # 为每个pattern只创建一个目录，使用原始pattern名称
        save_graph_dir_p = os.path.join(save_graph_dir, base_name)
        save_metadata_dir_p = os.path.join(save_metadata_dir, base_name)
        if not os.path.isdir(save_graph_dir_p):
            os.makedirs(save_graph_dir_p, exist_ok=True)
        if not os.path.isdir(save_metadata_dir_p):
            os.makedirs(save_metadata_dir_p, exist_ok=True)
        
        # 记录pattern统计信息
        pattern_stats = {
            'pattern_name': base_name,
            'vertices': pattern_vertices,
            'edges': pattern_edges,
            'vertex_labels': pattern_vertex_labels,
            'edge_labels': pattern_edge_labels,
            'graphs_generated': 0,
            'total_subgraphs_found': 0
        }
        
        for alpha in CONFIG["alphas"]:
            for number_of_graph_vertices in CONFIG["number_of_graph_vertices"]:
                if number_of_graph_vertices < pattern_vertices:
                    continue
                for number_of_graph_vertex_labels in CONFIG["number_of_graph_vertex_labels"]:
                    if number_of_graph_vertex_labels > number_of_graph_vertices:
                        continue
                    if number_of_graph_vertex_labels < pattern_vertex_labels:
                        continue
                    for number_of_graph_edges in CONFIG["number_of_graph_edges"]:
                        if number_of_graph_edges < number_of_graph_vertices - 1: # not connected
                            continue
                        if number_of_graph_edges > CONFIG["max_ratio_of_edges_vertices"] * number_of_graph_vertices: # too dense
                            continue
                        if number_of_graph_edges < pattern_edges:
                            continue
                        for number_of_graph_edge_labels in CONFIG["number_of_graph_edge_labels"]:
                            if number_of_graph_edge_labels > number_of_graph_edges:
                                continue
                            if number_of_graph_edge_labels < pattern_edge_labels:
                                continue
                            results.append(
                                pool.apply_async(generate_graphs, args=(
                                    graph_generator, number_of_graph_vertices, number_of_graph_edges,
                                    number_of_graph_vertex_labels, number_of_graph_edge_labels,
                                    alpha, CONFIG["max_pattern_counts"], CONFIG["max_subgraph"],
                                    CONFIG["number_of_graphs"], save_graph_dir_p, save_metadata_dir_p)))
                            graph_cnt += CONFIG["number_of_graphs"]
                            pattern_stats['graphs_generated'] += CONFIG["number_of_graphs"]
    
        total_subgraph_stats.append(pattern_stats)
    
    pool.close()
    success_count = 0
    error_count = 0
    for x in tqdm(results):
        try:
            x.get()
            success_count += 1
        except Exception as e:
            print(f"Error in graph generation task: {e}")
            error_count += 1
    print(f"{success_count} graphs generation finished successfully, {error_count} tasks failed!")
    
    # 生成总统计报告
    summary_file = os.path.join(CONFIG["save_data_dir"], "subgraph_generation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Subgraph Generation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Graphs Generated: {graph_cnt}\n")
        f.write(f"Number of Patterns: {len(existing_patterns)}\n\n")
        
        f.write("Pattern Details:\n")
        f.write("-" * 30 + "\n")
        for stats in total_subgraph_stats:
            f.write(f"Pattern: {stats['pattern_name']}\n")
            f.write(f"  Vertices: {stats['vertices']}, Edges: {stats['edges']}\n")
            f.write(f"  Vertex Labels: {stats['vertex_labels']}, Edge Labels: {stats['edge_labels']}\n")
            f.write(f"  Graphs Generated: {stats['graphs_generated']}\n")
            f.write("\n")
    
    print(f"Subgraph generation summary written to: {summary_file}")
