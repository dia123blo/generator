import numpy as np
import igraph as ig
import json
from itertools import chain, combinations
import random
import os

def generate_png(dot_filename, png_filename=None, prog="neato"):
    if png_filename is None:
        png_filename = dot_filename.replace(".dot", ".png")
    os.system("%s.exe -T png %s > %s" % (prog, dot_filename, png_filename))


def generate_vf_random_in_range(min_val, max_val):
    """
    生成符合电路节点编码规则的6位二进制数
    前3位：器件类型（0-5），后3位：类内编码（0-7）

    """
    device_type_list = [ ('1',20), ('2',20), ('3',1), ('4',2), ('5',1),('6',1)]
    device_type = random.randint(1, 6)


    device_choices = []
    for device, count in device_type_list:
        device_choices.extend([int(device)] * count)
    device_type = random.choice(device_choices)




    intra_class_code_list = ['0', '1']
    # 器件类型：nmos(1), pmos(2), 电容(3), 电阻(4), 二极管(5), BJT(6)

    
    
    
    # 类内编码：0-7，表示不同特性
    intra_class_code = random.randint(0, 1)
    
    # 组合成6位二进制数：前3位 + 后3位
    
    
    return int(str(device_type)+str(intra_class_code))

def generate_vf_random_in_range_list(number_of_items, min_val=0, max_val=63):
    """
    生成节点编码列表
    """
    return [generate_vf_random_in_range(min_val, max_val) for _ in range(number_of_items)]



def generate_edge_label_by_device_types(source_device, target_device, connection_type="default"):
    """
    根据器件类型生成完整的边编码
    完整格式：前3位（输出→输入）+ 后1位（大小关系）
    """
    # try:
    # 确保输入是整数或可转换为整数的字符串
    if isinstance(source_device, str):
        source_device = int(source_device)
    if isinstance(target_device, str):
        target_device = int(target_device)
        
    # 提取器件类型（前3位）
    source_type = str(source_device)[:1]
    target_type = str(target_device)[:1]
    
    # 定义权重对及其出现次数分布
    weight_pairs = [
        ((1, 1), 192),
        ((2, 8), 784),
        ((3, 9), 48),
        ((4, 64), 312),
        ((5, 65), 16),
        ((10, 10), 6),
        ((12, 66), 16),
        ((16, 16), 460),
        ((17, 17), 230),
        ((18, 24), 10),
        ((20, 80), 28),
        ((32, 128), 224),
        ((34, 136), 2),
        ((36, 192), 8),
        ((49, 145), 2),
        ((56, 146), 4),
        ((68, 68), 18),
        ((84, 84), 4),
        ((96, 132), 2),
        ((256, 256), 3672),
        ((257, 257), 110),
        ((258, 264), 328),
        ((259, 265), 6),
        ((260, 320), 556),
        ((266, 266), 14),
        ((268, 322), 2),
        ((272, 272), 474),
        ((273, 273), 2),
        ((274, 280), 60),
        ((276, 336), 6),
        ((284, 338), 4),
        ((288, 384), 22),
        ((290, 392), 2),
        ((292, 448), 136),
        ((325, 325), 32),
        ((341, 341), 182),
        ((365, 455), 30)
    ]
    
    # 根据出现次数创建权重选择列表
    weighted_choices = []
    for pair, count in weight_pairs:
        weighted_choices.extend([pair] * count)
    
    # 有源器件：nmos(0), pmos(1), BJT(5)
    active_devices = ['1', '2']  # 字符串形式
    # 无源器件：电容(2), 电阻(3), 二极管(4)
    passive_devices = ['3', '4', '5', '6']
    size_relation_options = {
        # 00为缺省值
    "input_less": '2', # 0F为输入端小于输出端
    "input_greater": '5', # F0为输入端大于输出端
    "input_equal": '9'   # FF为输入端等于输出端
}
    
    # 确定连接关系
    if source_type in active_devices and target_type in active_devices:
        # 有源器件间互联 - 根据权重分布选择权重对
        selected_pair = random.choice(weighted_choices)
        out_to_in, in_to_out = selected_pair
        
        if random.random() < 0.5:
            out_to_in = in_to_out
        # # 大小关系编码：正向大为9，反向大为1，相等为5
        # if out_to_in > in_to_out:
        #     size_relation = '9'  # 正向大
        # elif out_to_in < in_to_out:
        #     size_relation = '1'  # 反向大
        # else:
        #     size_relation = '5'  # 相等
        # 大小关系编码选项
#     

        # 随机选择大小关系（实际应用中应根据具体关系确定）
        size_relation = random.choice(list(size_relation_options.values()))
            
    elif source_type in active_devices and target_type in passive_devices:
        # 有源→无源 - 有源侧置F，无源侧置0
        out_to_in = 000  # 有源侧全连接
        size_relation = '0'  # 缺省值
        
    elif source_type in passive_devices and target_type in active_devices:
        # 无源→有源 - 无源侧置0，有源侧置F
        out_to_in = 000   # 无源侧置0
        size_relation = '0'  # 无源→有源时，大小关系置为缺省值
        
    else:
        # 无源器件间互联 - 输出到输入的权重为0
        out_to_in = 000
        size_relation = '0'
    
    # 组合成4位编码
    edge_code =  int(size_relation+str(out_to_in).zfill(3)) 
    
    # # 避免全0的编码
    # if edge_code == '0000':
    #     edge_code = '0015'  # 默认非零编码
        
    return edge_code





def generate_labels(number_of_items, number_of_labels):
    labels = list(range(number_of_labels))
    if number_of_items < number_of_labels:
        np.random.shuffle(labels)
        labels = labels[:number_of_items]
    else:
        for i in range(number_of_labels, number_of_items):
            labels.append(np.random.randint(number_of_labels))
        np.random.shuffle(labels)
    return labels

def generate_tree(number_of_vertices, directed=True):
    # Alexey S. Rodionov and Hyunseung Choo, On Generating Random Network Structures: Trees, ICCS 2003, LNCS 2658, pp. 879-887, 2003.
    # [connected vertices] + [unconnected vertices]
    shuffle_vertices = list(range(number_of_vertices))
    np.random.shuffle(shuffle_vertices)
    # randomly choose one vertex from the connected vertex set
    # randomly choose one vertex from the unconnected vertex set
    # connect them by one edge
    # add the latter vertex in the connected vertex set
    edges = list()
    for v in range(1, number_of_vertices):
        u = shuffle_vertices[np.random.randint(0, v)]
        v = shuffle_vertices[v]
        if get_direction():
            src_tgt = (u, v)
        else:
            src_tgt = (v, u)
        edges.append(src_tgt)
    tree = ig.Graph(directed=directed)
    tree.add_vertices(number_of_vertices)
    tree.add_edges(edges)
    return tree

def get_direction():
    return np.random.randint(0, 2)

def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)     
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)

def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes" or x == "t"

def sample_element(s):
    index = np.random.randint(0, len(s))
    return s[index]

def powerset(iterable, min_size=0, max_size=-1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = sorted(iterable)
    if max_size == -1:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size+1))