
import numpy as np
import torch
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import zoom
'''
1. 把原始 .mat 脑电数据提取、预处理、转成 .pth 格式（PyTorch 专用） 
2. 批量生成 / 加载每个被试、每个会话的数据，方便直接喂给模型训练

读取 SEED 脑电 .mat 数据
标准化预处理
按被试 / 会话保存成 .pth
批量生成 + 自动检查
最终输出可直接训练的干净数据集
'''

SEED_data_dir = "C:/Users/34012/Desktop/seed"
Data_root = os.path.join(SEED_data_dir, "dataset_seed_iv")   # 生成的 .pth 存放处

session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = [session1_label, session2_label, session3_label]

#标准化
def eeg_transformer(x, data_process="Standardization"):
    if data_process == "Standardization" :
        return (x - x.mean()) / x.std()
    # if data_process == "min_max_normalization":
    #     return (x - x.min()) / (x.max - x.min() + 1e-8)



# 从SEED-IV数据集中提取单个受试者（subject）在某个会话（session）中的数据，并将其保存为一个PyTorch张量文件（.pth）
def single_subject_dataset(session_num=1, subj_num=1):

    sample_list = []        #存储处理后的样本数据
    sample_label_list = []  #存储样本对应的标签

    # 4.1 找到原始 mat 文件
    session_dir = os.path.join(SEED_data_dir, "eeg_feature_smooth", str(session_num))
    for fname in os.listdir(session_dir):
        if int(fname.split("_")[0]) == subj_num: #按下划线切开变成列表，取第 0 号（第 1 个）元素，其实就是找到 第 subj_num 号受试者 的 .mat 数据文件
            mat_path = os.path.join(session_dir, fname)
            break
    else:
        raise FileNotFoundError(f"未找到受试者 {subj_num} 的 mat 文件！")

    # 4.2 提取数据
    for trial in range(24):
        de = scipy.io.loadmat(mat_path)[f'de_LDS{trial + 1}']  # (chan, time, freq) (62,58,5)
        y = int(labels[session_num - 1][trial])
        # 把每个时间片的 62×5 作为一个样本，送入模型训练
        for t in range(de.shape[1]): 
            x = torch.tensor(de[:, t, :]).float() # 取第 t 个时间片，形状 (62, 5)
            x = eeg_transformer(x, "Standardization")
            sample_list.append(x)
            sample_label_list.append(y)

    # 4.3 保存
    out_dir = os.path.join(Data_root, f"{session_num}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'data': sample_list, 'labels': sample_label_list},
               os.path.join(out_dir, f"subject{subj_num}.pth"))
    print(f"已保存 -> {out_dir}/subject{subj_num}.pth")


def load_subject(session_num=1, subj_num=1):
    pth_path = os.path.join(Data_root, f"{session_num}", f"subject{subj_num}.pth")
    if not os.path.exists(pth_path):
        print(f"路径不存在：{pth_path}")
        return None

    dic = torch.load(pth_path)
    data, label = dic['data'], dic['labels']
    print(f"subject: {subj_num}, samples_num: {len(data)}")
    uniq, cnt = np.unique(label, return_counts=True)
    print(dict(zip(uniq, cnt)))
    return label


if __name__ == "__main__":

    session_num = 3
    ref_label = load_subject(session_num=session_num, subj_num=1)
    for subj in range(1,16):
        pth_path =  os.path.join(Data_root, f"{session_num}", f"subject{subj}.pth")
        

        if not os.path.exists(pth_path):
            print(f"缺失 -> 正在生成：{pth_path}")
            single_subject_dataset(session_num, subj)
        else:
            lbl = load_subject(session_num, subj)
            if torch.equal(torch.tensor(ref_label), torch.tensor(lbl)):
                print(f"subject {subj} 标签与 reference 一致。")





