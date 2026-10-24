import numpy as np
import torch
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import zoom
from grain_ball import generate_grain_balls, get_grain_ball_representation

SEED_data_dir = "C:/Users/34012/Desktop/seed"
Data_root = os.path.join(SEED_data_dir, "dataset_seed")   # 生成的 .pth 存放处


seesion1_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
seesion2_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
seesion3_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
labels = [seesion1_label, seesion2_label, seesion3_label]

#定义训练数据集（14 subject）
class  SEED_dataset_train(Dataset):
    def __init__(self, subj_test, session_num):
        super(SEED_dataset_train, self).__init__()

        self.subj_test = subj_test
        self.session_num = session_num
        self.sample_lists = [[] for i in range(14)]
        self.sample_label_lists = [[] for i in range(14)]
        self.sample_list = []
        self.sample_label_list = []
        self.subj_id_list = []          # 每个样本对应的 0~13 编号
        # 记录“被试编号 → 在 14 人里的索引”


        i = 0
        for subj_num in range(1, 16):
            
            if subj_num == subj_test: 
                continue
            idx = i
            i = i + 1
                
            path = os.path.join(Data_root, f"{session_num}", f"subject{subj_num}.pth")
            dataset_dict = torch.load(path)

            data = dataset_dict['data']
            labels = dataset_dict['labels']
            self.sample_lists[i-1] = dataset_dict['data'] # 共享相同的引用。
            self.sample_label_lists[i-1] = dataset_dict['labels']
            # 关键：把当前人的编号重复 len(data) 次
            self.subj_id_list.extend([idx] * len(data))

        for i in range(14):
            self.sample_list.extend(self.sample_lists[i])
            self.sample_label_list.extend(self.sample_label_lists[i])
            
        print(len(self.sample_label_list))

    def __len__(self, ):

        return len(self.sample_list)

    def __getitem__(self, index):
        
        return self.sample_list[index], self.sample_label_list[index],self.subj_id_list[index] 


# 定义测试数据集（1 subject）
class  SEED_dataset_test(Dataset):
    def __init__(self, subj_test, session_num):
        super(SEED_dataset_test, self).__init__()

        self.subj_test = subj_test
        self.session_num = session_num
        self.sample_lists = []
        self.sample_label_list = []
        path = os.path.join(Data_root, f"{session_num}", f"subject{subj_test}.pth")
        dataset_dict = torch.load(path)
        self.sample_lists = dataset_dict['data'] 
        self.sample_label_list = dataset_dict['labels']
        self.subj_id_list = [0] * len(self.sample_label_list)

    def __len__(self, ):
        return len(self.sample_lists)
                                  
    def __getitem__(self, index):
        return self.sample_lists[index], self.sample_label_list[index],self.subj_id_list[index]


def getloader(subj_test=1, session_num=1, batch_size=8, is_drop_last=False, is_shuffle=True, is_train=True):
    if is_train:
        gen_dataset = SEED_dataset_train(subj_test=subj_test, session_num=session_num)
        for subj_idx, (data, label) in enumerate(gen_dataset.iter_subj()):
            # data  : 该人全部脑电样本（list 或 tensor）
            # label : 对应标签
            print(f'Person {subj_idx}: {len(label)} samples')

    else:
        gen_dataset = SEED_dataset_test(subj_test=subj_test, session_num=session_num)

    return DataLoader(
            gen_dataset,
            batch_size=batch_size,
            shuffle = is_shuffle, 
            num_workers=0, 
            drop_last=is_drop_last,
        )

 

# 定义训练数据集（14 subject）带粒球聚合
class SEED_dataset_train_with_GB(Dataset):
    def __init__(self, subj_test, session_num, target_num_samples=None):
        super(SEED_dataset_train_with_GB, self).__init__()

        self.subj_test = subj_test
        self.session_num = session_num
        
        # 1. 加载源域数据 (14 subjects，排除测试subject)
        self.source_data = []
        self.source_labels = []
        
        i = 0
        for subj_num in range(1, 16):
            if subj_num == subj_test:
                continue
            path = os.path.join(Data_root, f"{session_num}", f"subject{subj_num}.pth")
            dataset_dict = torch.load(path)
            data = dataset_dict['data']
            labels = dataset_dict['labels']
            
            self.source_data.extend(data)
            self.source_labels.extend(labels)
            i += 1
        
        print(f"源域样本数: {len(self.source_labels)}")
        
        # 2. 获取目标域样本数 (用于确定粒球数量)
        if target_num_samples is None:
            target_path = os.path.join(Data_root, f"{session_num}", f"subject{subj_test}.pth")
            target_dict = torch.load(target_path)
            self.target_num_samples = len(target_dict['data'])
        else:
            self.target_num_samples = target_num_samples
        
        print(f"目标域样本数 M = {self.target_num_samples}")
        
        # 3. 检查是否已有保存的粒球
        gb_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grain_balls_saved")
        gb_path = os.path.join(gb_save_dir, f"grain_balls_subj{subj_test}_session{session_num}.pth")

        if os.path.exists(gb_path):
            print("加载已保存的粒球...")
            self.grain_balls = torch.load(gb_path)
        else:
            source_tensor = torch.stack(self.source_data)
            print("开始生成粒球...")
            self.grain_balls = generate_grain_balls(
                source_tensor, 
                self.source_labels, 
                target_num=self.target_num_samples,
                tolerance=0.2
            )
            os.makedirs(gb_save_dir, exist_ok=True)
            torch.save(self.grain_balls, gb_path)
            print(f"粒球已保存至: {gb_path}")

        # 4. 提取粒球中心、标签和半径
        self.gb_centers, self.gb_labels, self.gb_radii = get_grain_ball_representation(self.grain_balls)

        print(f"粒球数 K = {len(self.grain_balls)}")
        print(f"K ≈ M: {abs(len(self.grain_balls) - self.target_num_samples) <= self.target_num_samples * 0.2}")


    def __len__(self):
        # 只返回粒球数 (源域)，目标域通过dataloader_target提供
        return len(self.gb_centers)
    
    def __getitem__(self, index):
        return self.gb_centers[index], self.gb_labels[index], self.gb_radii[index], torch.tensor(0, dtype=torch.long)


if __name__== "__main__":

    dataloader_train = getloader(subj_test=2, session_num=2, batch_size=32, is_drop_last=False, is_shuffle=True, is_train=True)
    dataloader_test = getloader(subj_test=2, session_num=1, batch_size=32, is_drop_last=False, is_shuffle=True, is_train=False)

    for step, (data_samples, labels) in enumerate(dataloader_train):
        batch_data = data_samples           # -> [14*batch_size, feat_dim]
        batch_labels = labels                      # -> [14*batch_size]
        print(data_samples[0].shape, step)
        print(batch_data.shape)
        # print(labels)

    print("test")

