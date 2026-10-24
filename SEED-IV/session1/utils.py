import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 统计预测标签的预测稳定性
class LabelStabilityTracker: # 这个是有问题的，因为数据加载的时候是随机采样的，因此就算每次都预测成功，但标签的顺序也是不一样的
    def __init__(self, stability_threshold=0.05, stable_count_threshold=3):
        """
        初始化稳定性追踪器
        
        参数:
        stability_threshold (float): 标签变化率阈值，低于该值视为稳定（如 0.05 表示 5%）
        stable_count_threshold (int): 连续稳定的次数阈值，用于判定稳定（如 3 表示连续 3 轮稳定）
        """
       	self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prev_labels = None
        self.stability_threshold = stability_threshold
        self.stable_count_threshold = stable_count_threshold
        self.stable_count = 0
    
    def compute_label_change_rate(self, current_labels):
        """
        计算当前标签相对于上一轮的变化率
        
        参数:
        current_labels (torch.Tensor): 当前轮的预测标签，形状为 [num_samples]
        
        返回:
        float: 标签变化率，表示前后两轮标签不一致的比例
        """
        if self.prev_labels is None:
            # 第一轮时没有上一轮标签，变化率设为 0
            change_rate = 0.0
        else:
            num_changes = torch.sum(self.prev_labels != current_labels).item()
            change_rate = num_changes / len(current_labels) 
                    
        # 更新上一轮标签
        self.prev_labels = current_labels.clone()
        
        # 判断是否达到稳定性
        if change_rate < self.stability_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        # 返回变化率
        return change_rate
    
    def is_stable(self):
        """
        检查是否达到连续稳定的阈值，用于停止条件
        
        返回:
        bool: 若达到连续稳定的次数阈值，返回 True；否则返回 False
        """
        print(self.stable_count, self.stable_count_threshold)
        return self.stable_count >= self.stable_count_threshold

    def track_stability(self, model, test_loader):
        """
        计算并返回标签变化率
        
        参数:
        model (torch.nn.Module): 训练中的模型
        test_loader (DataLoader): 测试集的数据加载器
        
        返回:
        float: 当前轮标签变化率
        """
        model.eval()  # 设置模型为评估模式
        current_labels = []
        
        acc = 0
        tes_acc = 0
        with torch.no_grad():
            for data, labels in test_loader:
                inputs = data.to(self.device)  # 确保数据和模型在同一设备
                outputs, _ = model(inputs)
                preds = outputs.argmax(dim=1)  # 获取预测标签
                current_labels.append(preds)

                # 预测准确率
                tes_acc += (preds == labels.to(self.device)).sum().item()
        tes_acc = tes_acc / len(test_loader.dataset)

        current_labels = torch.cat(current_labels)
        change_rate = self.compute_label_change_rate(current_labels)
        
        # 回到训练模式
        model.train()
        
        return change_rate, tes_acc



# cite from "https://github.com/jindongwang/transferlearning/blob/master/code/DeepDA/loss_funcs/mmd.py"
class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    # 把 [N,62,5] 脑电特征映射到一个高维空间
    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss