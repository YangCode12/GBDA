import torch
import numpy as np
from collections import Counter

# ---------------------- GBSC 核心工具函数（适配PyTorch张量） ----------------------
def gbsc_get_radius(gb_data):
    """GBSC风格计算粒球半径（最大距离，适配[62,5]形状张量）"""
    if len(gb_data) == 0:
        return 0.0
    center = gb_data.mean(dim=0)  # [62,5]
    distances = torch.norm(gb_data - center, dim=(1, 2))  # 每个样本到中心的距离，shape [n]
    return distances.max().item()

def gbsc_get_density(gb_data):
    """GBSC风格计算粒球密度：点数 / 到中心距离之和"""
    num = len(gb_data)
    if num == 0:
        return 0.0
    center = gb_data.mean(dim=0)
    distances = torch.norm(gb_data - center, dim=(1, 2))
    sum_radius = distances.sum().item() 
    return num / sum_radius if sum_radius != 0 else num

def gbsc_spilt_ball_2(gb_data, sample_indices, ball_labels):
    """
    GBSC核心分裂方法：最远点对分裂，适配[62,5]形状张量
    找到距离最远的两个样本 (r1, c1) 作为两个初始中心，将其他样本分配到离它们更近的中心
    返回：(子球1数据, 子球2数据, 子球1索引, 子球2索引, 子球1标签, 子球2标签)，无法分裂返回None
    """
    n = len(gb_data)
    if n < 2:
        return None, None, None, None, None, None
    
    # 1. 展平样本计算距离矩阵（[62,5] -> [310]）
    flat_data = gb_data.reshape(n, -1)  # [n, 310]
    dot = torch.matmul(flat_data, flat_data.T)  # 点积矩阵 [n,n]
    norm_sq = torch.diag(dot)  # 每个样本的平方范数 [n]
    D = torch.sqrt(torch.clamp(norm_sq.unsqueeze(0) + norm_sq.unsqueeze(1) - 2 * dot, min=0))  # 距离矩阵 [n,n]
    
    # 2. 找最远点对
    D.fill_diagonal_(0)
    max_val = D.max()
    if max_val == 0:  
        split = torch.randperm(n) < n // 2 # 所有点重合，随机划分
    else:
        r1, c1 = torch.where(D == max_val)
        r1, c1 = r1[0].item(), c1[0].item() # 把索引从张量格式，转成普通数字
        split = D[:, r1] < D[:, c1]  # 分配到距离更近的中心
    
    # 3. 分割结果
    b1_data = gb_data[split]
    b2_data = gb_data[~split]
    b1_idx = [sample_indices[i] for i in range(n) if split[i]]
    b2_idx = [sample_indices[i] for i in range(n) if not split[i]]
    b1_lab = [ball_labels[i] for i in range(n) if split[i]]
    b2_lab = [ball_labels[i] for i in range(n) if not split[i]]
    return b1_data, b2_data, b1_idx, b2_idx, b1_lab, b2_lab

def gbsc_check_split(parent_data, child1_data, child2_data):
    """GBSC分裂条件：子球加权密度之和 > 父球密度"""
    parent_density = gbsc_get_density(parent_data)
    d1 = gbsc_get_density(child1_data)
    d2 = gbsc_get_density(child2_data)
    total = len(child1_data) + len(child2_data)
    if total == 0:
        return False
    w1 = len(child1_data) / total
    w2 = len(child2_data) / total
    return (w1 * d1 + w2 * d2) > parent_density

class GrainBall:
    """粒球类：表示一个粒球"""
    def __init__(self, center, radius, sample_indices, label):
        self.center = center          # 形状: [62, 5]，粒球中心
        self.radius = radius          # float，半径(到中心最远距离)
        self.sample_indices = sample_indices  # list，粒球内样本索引（知道这个球包含哪些原始数据）
        self.label = label         # int，多数票标签

    #打印粒球信息
    def __repr__(self):
        return f"GrainBall(center_shape={self.center.shape}, radius={self.radius:.4f}, samples={len(self.sample_indices)}, label={self.label})"


def _make_sub_gb(sub_data, sub_idx, sub_lab, parent_gb):
    """构造子粒球的通用函数"""
    if len(sub_idx) == 0:
        return None
    c = sub_data.mean(dim=0)
    r = gbsc_get_radius(sub_data)
    lab = Counter(sub_lab).most_common(1)[0][0] if sub_lab else parent_gb.label
    return GrainBall(c, r, sub_idx, lab)

def generate_grain_balls(data, labels, target_num, tolerance=0.2):
    """
    粒球生成主函数 - GBSC风格分裂 + 原数量调整逻辑
    
    输入:
        data: 源域数据 [N, 62, 5] 
        labels: 源域标签 [N] 
        target_num: 目标粒球数 (≈目标域样本数)
        tolerance: 允许误差比例，默认20%
    
    输出:
        grain_balls: 粒球列表，数量在 [target_num*(1-tolerance), target_num*(1+tolerance)] 范围内
    """
    n = len(data)
    if n == 0:
        return []
    
    # 确保labels是list
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    
    # Step 1: 初始化 - 整个数据集作为1个粒球
    all_indices = list(range(n))
    center = data.mean(dim=0)  # [62, 5]
    
    # 计算半径（使用GBSC方法）
    radius = gbsc_get_radius(data)
    
    # 多数票标签
    majority_label = Counter([labels[i] for i in all_indices]).most_common(1)[0][0]
    
    initial_ball = GrainBall(center, radius, all_indices, majority_label)
    grain_balls = [initial_ball]
    
    min_target = int(target_num * (1 - tolerance))
    max_target = int(target_num * (1 + tolerance))
    print(f"目标粒球数范围: [{min_target}, {max_target}]")
    
    # ---------------------- 第一轮：GBSC密度条件分裂（内置数量上限） ----------------------
    while True:
        # 提前退出：粒球数量已达目标上限
        if len(grain_balls) >= max_target:
            print(f"第一轮密度分裂已达上限，当前粒球数: {len(grain_balls)}")
            break
        new_balls = []
        split_flag = False # 标签是否分裂
        for gb in grain_balls:
            # GBSC阈值：粒球点数<8不分裂
            if len(gb.sample_indices) < 8:
                new_balls.append(gb)
                continue
            
            ball_data = data[gb.sample_indices]  # [n_local, 62, 5]
            ball_labels = [labels[i] for i in gb.sample_indices]
            
            # GBSC最远点对分裂
            res = gbsc_spilt_ball_2(ball_data, gb.sample_indices, ball_labels)
            if res[0] is None:
                new_balls.append(gb)
                continue
            
            b1_data, b2_data, b1_idx, b2_idx, b1_lab, b2_lab = res
            
            # 检查分裂条件：子球加权密度之和 > 父球密度
            if gbsc_check_split(ball_data, b1_data, b2_data):
                sub1 = _make_sub_gb(b1_data, b1_idx, b1_lab, gb)
                sub2 = _make_sub_gb(b2_data, b2_idx, b2_lab, gb)
                if sub1:
                    new_balls.append(sub1)
                if sub2:
                    new_balls.append(sub2)
                split_flag = True
            else:
                new_balls.append(gb)
        
        if not split_flag: # 有过分裂，则粒球列表需要改变，没有则粒球等于原来的
            break
        grain_balls = new_balls
    
    # ---------------------- 第二轮：GBSC半径归一化分裂 ----------------------
    radii = [gb.radius for gb in grain_balls if len(gb.sample_indices) >= 2] # 所有样本>2的粒球的半径
    radius_detect = max(np.median(radii), np.mean(radii)) if radii else 0.0
    
    while True:
        new_balls = []
        split_flag = False
        for gb in grain_balls:
            if len(gb.sample_indices) < 2 or gb.radius <= 2 * radius_detect:
                new_balls.append(gb)
                continue
            
            ball_data = data[gb.sample_indices]
            ball_labels = [labels[i] for i in gb.sample_indices]
            res = gbsc_spilt_ball_2(ball_data, gb.sample_indices, ball_labels)
            if res[0] is None:
                new_balls.append(gb)
                continue
            
            b1_data, b2_data, b1_idx, b2_idx, b1_lab, b2_lab = res
            sub1 = _make_sub_gb(b1_data, b1_idx, b1_lab, gb)
            sub2 = _make_sub_gb(b2_data, b2_idx, b2_lab, gb)
            if sub1:
                new_balls.append(sub1)
            if sub2:
                new_balls.append(sub2)
            split_flag = True
        
        if not split_flag:
            break
        grain_balls = new_balls
    print(f"第二轮粒球数: {len(grain_balls)}")

    # ---------------------- 数量不足：分裂最大粒球（GBSC方式） ----------------------
    while len(grain_balls) < min_target:
        grain_balls.sort(key=lambda gb: gb.radius, reverse=True)
        largest_ball = grain_balls.pop(0)
        
        if len(largest_ball.sample_indices) < 2:
            grain_balls.append(largest_ball)
            break
        
        ball_data = data[largest_ball.sample_indices]
        ball_labels = [labels[i] for i in largest_ball.sample_indices]
        res = gbsc_spilt_ball_2(ball_data, largest_ball.sample_indices, ball_labels)
        if res[0] is None:
            grain_balls.append(largest_ball)
            break
        
        b1_data, b2_data, b1_idx, b2_idx, b1_lab, b2_lab = res
        sub1 = _make_sub_gb(b1_data, b1_idx, b1_lab, largest_ball)
        sub2 = _make_sub_gb(b2_data, b2_idx, b2_lab, largest_ball)
        if sub1:
            grain_balls.append(sub1)
        if sub2:
            grain_balls.append(sub2)
    
    # ---------------------- 数量超标：合并最小粒球（原逻辑保留） ----------------------
    # 合并时，在标签相同的粒球中，找中心距离最近的两个合并
    while len(grain_balls) > max_target:
        # 按标签分组
        label_to_balls = {}
        for gb in grain_balls:
            label_to_balls.setdefault(gb.label, []).append(gb)   

        best_pair = None
        best_dist = float('inf')
        for label, balls in label_to_balls.items():
            if len(balls) < 2:
                continue
            centers = torch.stack([b.center.reshape(-1) for b in balls])
            dist_mat = torch.cdist(centers, centers)
            for i in range(len(balls)):
                for j in range(i+1, len(balls)):
                    d = dist_mat[i, j].item()
                    if d < best_dist:
                        best_dist = d
                        best_pair = (balls[i], balls[j])

        if best_pair is None:           # 所有标签都只有一个粒球，退化为原策略
            grain_balls.sort(key=lambda gb: len(gb.sample_indices))
            gb1 = grain_balls.pop(0)    # 安全移除
            gb2 = grain_balls.pop(0)
        else:
            gb1, gb2 = best_pair
            grain_balls.remove(gb1)
            grain_balls.remove(gb2)

        # 执行合并
        merged_indices = gb1.sample_indices + gb2.sample_indices
        merged_data = data[merged_indices]
        merged_center = merged_data.mean(dim=0)
        merged_radius = gbsc_get_radius(merged_data) if len(merged_indices) > 1 else 0.0
        merged_label = gb1.label if gb1.label == gb2.label else Counter(
            [labels[i] for i in merged_indices]).most_common(1)[0][0]

        merged_ball = GrainBall(merged_center, merged_radius, merged_indices, merged_label)
        grain_balls.append(merged_ball)

    return grain_balls


def get_grain_ball_representation(grain_balls):
    """
    把粒球列表转换成模型能用的张量，从粒球列表提取中心、标签和半径
    
    输入:
        grain_balls: 粒球列表
    
    输出:
        centers: [K, 62, 5] 粒球中心 (torch.Tensor)
        labels: [K] 多数票标签 (torch.Tensor)
        radii: [K] 粒球半径 (torch.Tensor)
    """
    if not grain_balls:
        return torch.empty(0, 62, 5), torch.empty(0, dtype=torch.long), torch.empty(0)
    
    centers = torch.stack([gb.center for gb in grain_balls])
    labels = torch.tensor([gb.label for gb in grain_balls], dtype=torch.long)
    radii = torch.tensor([gb.radius for gb in grain_balls])
    return centers, labels, radii