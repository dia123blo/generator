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
    # 器件类型：nmos(0), pmos(1), 电容(2), 电阻(3), 二极管(4), BJT(5)
    device_type = random.randint(0, 5)
    
    # 类内编码：0-7，表示不同特性
    intra_class_code = random.randint(0, 7)
    
    # 组合成6位二进制数：前3位 + 后3位
    node_code = (device_type << 3) | intra_class_code
    
    return node_code

def generate_vf_random_in_range_list(number_of_items, min_val=0, max_val=63):
    """
    生成节点编码列表
    """
    return [generate_vf_random_in_range(min_val, max_val) for _ in range(number_of_items)]


def generate_ef_random_in_range(min_val, max_val):
    """
    生成符合电路边编码规则的8位十六进制数
    格式：前3位(输出→输入) + 中3位(输入→输出) + 后2位(大小关系)
    """
    # # 方向连接情况（随机生成，实际应用中应根据具体连接关系确定）
    # out_to_in = random.randint(0, 511)  # 12位，用3位十六进制表示
    # in_to_out = random.randint(0, 511)  # 12位，用3位十六进制表示


     # 连接类型编码值
    connection_codes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    
    
    # 确定连接关系
    
    # 有源器件间互联 - 两个方向都使用连接编码
    num_connections_out_in = random.randint(1, 4)
    selected_codes_out_in = random.sample(connection_codes, num_connections_out_in)
    out_to_in = sum(selected_codes_out_in)
    
    num_connections_in_out = random.randint(1, 4)
    selected_codes_in_out = random.sample(connection_codes, num_connections_in_out)
    in_to_out = sum(selected_codes_in_out)
    
    # 大小关系编码
    size_relation_options = [ '09', '90', '99']
    size_relation = random.choice(size_relation_options)
    
    # 组合成8位十六进制数
    edge_code = str(out_to_in).zfill(3) + str(in_to_out).zfill(3) + size_relation

    if edge_code =='0':
        pass
    
    return edge_code

def generate_ef_random_in_range_list(number_of_items, min_val=0, max_val=99999999):
    """
    生成边编码列表
    """
    return [generate_ef_random_in_range(min_val, max_val) for _ in range(number_of_items)]

def generate_edge_label_by_device_types(source_device, target_device, connection_type="default"):
    """
    根据器件类型生成完整的边编码
    完整格式：前12位（输出→输入）+ 中12位（输入→输出）+ 后8位（大小关系）
    """
    # 提取器件类型（前3位）
    source_type = (source_device >> 3) & 0x7
    target_type = (target_device >> 3) & 0x7
    
    # 连接类型编码值
    connection_codes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # 大小关系编码选项
    size_relation_options = {
            # 00为缺省值
        "input_less": '09', # 0F为输入端小于输出端
        "input_greater": '90', # F0为输入端大于输出端
        "input_equal": '99'   # FF为输入端等于输出端
    }
    
    # 随机选择大小关系（实际应用中应根据具体关系确定）
    size_relation = random.choice(list(size_relation_options.values()))
    
    # 有源器件：nmos(0), pmos(1), BJT(5)
    active_devices = [0, 1]
    # 无源器件：电容(2), 电阻(3), 二极管(4)
    passive_devices = [2, 3, 4,5]
    
    # 确定连接关系
    if source_type in active_devices and target_type in active_devices:
        # 有源器件间互联 - 两个方向都使用连接编码
        num_connections_out_in = random.randint(1, 3)
        selected_codes_out_in = random.sample(connection_codes, num_connections_out_in)
        out_to_in = sum(selected_codes_out_in)
        
        num_connections_in_out = random.randint(1, 3)
        selected_codes_in_out = random.sample(connection_codes, num_connections_in_out)
        in_to_out = sum(selected_codes_in_out)
        
    elif source_type in active_devices and target_type in passive_devices:
        # 有源→无源 - 有源侧置F，无源侧置0
        out_to_in = 999  # 有源侧全连接
        in_to_out = 000  # 无源侧置0
        
    elif source_type in passive_devices and target_type in active_devices:
        # 无源→有源 - 无源侧置0，有源侧置F
        out_to_in = 000  # 无源侧置0
        in_to_out = 999  # 有源侧全连接
        size_relation = '00'  # 无源→有源时，大小关系置为缺省值
        
    else:
        # 无源器件间互联 - 两个方向都使用较小的连接值范围
        # num_connections_out_in = random.randint(1, 2)
        # selected_codes_out_in = random.sample(connection_codes[:4], num_connections_out_in)
        out_to_in = 000
        
        # num_connections_in_out = random.randint(1, 2)
        # selected_codes_in_out = random.sample(connection_codes[:4], num_connections_in_out)
        in_to_out = 000
        size_relation = '00'
    
    # 组合成32位编码
    edge_code = str(out_to_in).zfill(3) + str(in_to_out).zfill(3) + size_relation
    return edge_code

def decode_edge_label(edge_code):
    """
    解析边编码，返回连接值、十六进制表示和使用的连接类型
    """
    # 提取各部分
    out_to_in = (edge_code >> 20) & 0xFFF
    in_to_out = (edge_code >> 8) & 0xFFF
    size_relation = edge_code & 0xFF
    
    # 计算使用的连接类型
    connection_codes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    used_codes = []
    remaining_value = out_to_in
    
    for code in sorted(connection_codes, reverse=True):
        if remaining_value >= code:
            used_codes.append(code)
            remaining_value -= code
    
    # 转换为十六进制
    connection_hex = hex(out_to_in)[2:].upper().zfill(3)
    
    return {
        "connection_value": out_to_in,
        "hex_representation": connection_hex,
        "used_codes": used_codes,
        "out_to_in": out_to_in,
        "in_to_out": in_to_out,
        "size_relation": size_relation
    }
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


# ===== Device Parameterization (64-dim) =====
DEVICE_TYPE_ENCODING = {
    "NMOS": 0,
    "PMOS": 1,
    "INDUCTOR": 2,
    "CAPACITOR": 3,
    "RESISTOR": 4,
    "DIODE": 5,
    "BJT": 6,
}

VOLTAGE_DOMAIN_ENCODING = {
    "HVD": 0,
    "CVD": 1,
    "LVD": 2,
}

# 单位: 米
TECH_NODE_LIMITS = {
    "110nm": {
        "min_W": 110e-9,
        "min_L": 110e-9,
        "max_W": 100e-6,
        "max_L": 10e-6,
    },
    "180nm": {
        "min_W": 180e-9,
        "min_L": 180e-9,
        "max_W": 100e-6,
        "max_L": 10e-6,
    },
}


def _normalized(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return float((value - min_val) / (max_val - min_val))


def build_device_feature_vector(
    device_type,
    voltage_domain,
    width,
    length,
    multiplier,
    fingers,
    tech_node="180nm",
):
    """
    构建64维器件级嵌入向量:
    [0:7]   设备类型one-hot
    [7:10]  电压域one-hot
    [10:14] 连续尺寸特征(归一化W/L/M/F)
    [14:64] 预留维度(默认0)
    """
    if tech_node not in TECH_NODE_LIMITS:
        raise ValueError("Unsupported tech_node: %s" % tech_node)

    if device_type not in DEVICE_TYPE_ENCODING:
        raise ValueError("Unsupported device_type: %s" % device_type)

    if voltage_domain not in VOLTAGE_DOMAIN_ENCODING:
        raise ValueError("Unsupported voltage_domain: %s" % voltage_domain)

    limits = TECH_NODE_LIMITS[tech_node]
    min_w, max_w = limits["min_W"], limits["max_W"]
    min_l, max_l = limits["min_L"], limits["max_L"]

    # 严格遵守工艺最小W/L约束
    width = max(width, min_w)
    length = max(length, min_l)
    width = min(width, max_w)
    length = min(length, max_l)

    multiplier = max(1, int(multiplier))
    fingers = max(1, int(fingers))

    h = [0.0] * 64

    # 离散编码
    h[DEVICE_TYPE_ENCODING[device_type]] = 1.0
    h[7 + VOLTAGE_DOMAIN_ENCODING[voltage_domain]] = 1.0

    # 连续尺寸特征
    h[10] = _normalized(width, min_w, max_w)
    h[11] = _normalized(length, min_l, max_l)
    h[12] = _normalized(multiplier, 1, 128)
    h[13] = _normalized(fingers, 1, 128)

    return h


def sample_device_parameterization(tech_node="180nm"):
    """
    采样单个器件参数并返回64维特征向量及原始参数.
    """
    if tech_node not in TECH_NODE_LIMITS:
        raise ValueError("Unsupported tech_node: %s" % tech_node)

    limits = TECH_NODE_LIMITS[tech_node]
    device_type = random.choice(list(DEVICE_TYPE_ENCODING.keys()))
    voltage_domain = random.choice(list(VOLTAGE_DOMAIN_ENCODING.keys()))
    width = random.uniform(limits["min_W"], limits["max_W"])
    length = random.uniform(limits["min_L"], limits["max_L"])
    multiplier = random.randint(1, 128)
    fingers = random.randint(1, 128)

    feature_vector = build_device_feature_vector(
        device_type=device_type,
        voltage_domain=voltage_domain,
        width=width,
        length=length,
        multiplier=multiplier,
        fingers=fingers,
        tech_node=tech_node,
    )

    return {
        "device_type": device_type,
        "voltage_domain": voltage_domain,
        "W": width,
        "L": length,
        "Multiplier": multiplier,
        "Number_of_Fingers": fingers,
        "h_vD_0": feature_vector,
    }


def sample_device_parameterization_list(number_of_devices, tech_node="180nm"):
    """
    批量采样器件参数表征.
    """
    return [sample_device_parameterization(tech_node=tech_node) for _ in range(number_of_devices)]
