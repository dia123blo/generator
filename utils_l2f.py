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
    在指定范围内生成符合规则的随机十进制数
    """
    # 定义所有可能的区间（每个独热码对应的范围）
    ranges = [
        (64, 79),    # 0001 00 xxxx (64-79)
        (128, 143),  # 0010 00 xxxx (128-143)
        (256, 271),  # 0100 00 xxxx (256-271)
        (512, 527)   # 1000 00 xxxx (512-527)
    ]
    
    # 计算有效区间（与用户范围的交集）
    valid_ranges = []
    for r_low, r_high in ranges:
        low = max(r_low, min_val)
        high = min(r_high, max_val)
        if low <= high:
            valid_ranges.append((low, high))
    
    if not valid_ranges:
        raise ValueError("指定范围内无有效数值")
    
    # 随机选择一个有效区间
    chosen_range = random.choice(valid_ranges)
    
    # 在选中的区间内生成随机数
    return random.randint(chosen_range[0], chosen_range[1])


def generate_ef_random_in_range(min_val, max_val):
    """
    在指定范围内生成符合规则的随机十进制数
    """
    # 定义所有可能的区间（每个独热码对应的范围）
    ranges = [
        (64, 127),    # 0001 xxxxxx (64-79)
        (128, 191),  # 0010 xxxxxx (128-143)
        (256, 319),  # 0100 xxxxxx (256-271)

    ]
    
    # 计算有效区间（与用户范围的交集）
    valid_ranges = []
    for r_low, r_high in ranges:
        low = max(r_low, min_val)
        high = min(r_high, max_val)
        if low <= high:
            valid_ranges.append((low, high))
    
    if not valid_ranges:
        raise ValueError("指定范围内无有效数值")
    
    # 随机选择一个有效区间
    chosen_range = random.choice(valid_ranges)
    
    # 在选中的区间内生成随机数
    return random.randint(chosen_range[0], chosen_range[1])



def generate_vf_random_in_range_list(number_of_items, min_val=64, max_val=527):
    """
    根据number_of_items生成在默认范围内的vf随机数列表
    """
    return [generate_vf_random_in_range(min_val, max_val) for _ in range(number_of_items)]

def generate_ef_random_in_range_list(number_of_items, min_val=64, max_val=319):
    """
    根据number_of_items生成在默认范围内的ef随机数列表
    """
    return [generate_ef_random_in_range(min_val, max_val) for _ in range(number_of_items)]


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