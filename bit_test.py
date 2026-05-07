import random

def bin10_to_dec(bin_str):
    """
    将10位二进制字符串（前4位独热码，中间2位00，后4位任意）转换为十进制数
    """
    if len(bin_str) != 10:
        raise ValueError("必须为10位二进制数")
    
    # 验证前4位是否为独热码
    first_four = bin_str[:4]
    if first_four.count('1') != 1:
        raise ValueError("前4位必须有且只有一个1")
    
    # 验证中间2位是否为00
    if bin_str[4:6] != "00":
        raise ValueError("中间2位必须为00")
    
    return int(bin_str, 2)

def generate_random_in_range(min_val, max_val):
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

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 示例1：二进制转十进制
    binary_str = "0001001111"  # 前4位:0001, 中间:00, 后4位:1111(15)
    decimal_val = bin10_to_dec(binary_str)
    print(f"二进制 {binary_str} → 十进制 {decimal_val} (应为79)")
    
    # 示例2：在指定范围[70, 140]内生成随机数
    # 有效子范围: [70,79] (64-79区间) 和 [128,140] (128-143区间)
    random_val = generate_random_in_range(70, 140)
    print(f"[70,140]范围内的随机值: {random_val}")
    
    # 查看随机值的二进制表示
    bin_rep = bin(random_val)[2:].zfill(10)
    print(f"二进制表示: {bin_rep}")
    print(f"结构验证: 前4位={bin_rep[:4]} 中间2位={bin_rep[4:6]} 后4位={bin_rep[6:]}")