import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Main_model import *
from Main_train import *



plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Step 1. 初始化模型
# -----------------------------
model = model(feature_dim=5, hidden_dim=5, source_num=14, classs_num=3).to(device)

model.eval()

# -----------------------------
# Step 2. 合并所有受试者的测试特征
# -----------------------------
features_all, labels_all = [], []

for subj_num in range(1,2):  # 共15个受试者
    ckpt_path = f'./Pro_log/best_ACC_subj{subj_num}_session1.pth'
    print(f'Loading weights: {ckpt_path}')
    # 直接加载模型参数
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader_target = getloader(subj_test=subj_num, session_num=1, batch_size=128 , is_drop_last=False, is_shuffle=False, is_train=False)



    with torch.no_grad():
        for data, labels,subj_t in dataloader_target:
            data = data.to(device)
           
            feats = model.extract_feature(data)

         

            # x_fft 是您之前经过 torch.fft.fft() 得到的复数张量
            x_time = feats  # feats 已经是时域特征
          
            features_all.append(x_time.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

features = np.concatenate(features_all, axis=0)
labels = np.concatenate(labels_all, axis=0)

print(f'Extracted total features: {features.shape}, labels: {labels.shape}')

# -----------------------------
# Step 3. t-SNE 降维
# -----------------------------
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
features_2d = tsne.fit_transform(features)

# -----------------------------
# Step 4. 绘图
# -----------------------------
plt.figure(figsize=(6,4))
colors = [
    "#FFB347",  # Positive (亮杏黄)
    "#66CDAA",  # Neutral (薄荷绿)
    "#3A7CA5"   # Negative (湖蓝)
]



class_names = ['Positive', 'Neutral', 'Negative']  # 请根据标签对应修改

for i, cls in enumerate(np.unique(labels)):
    idx = labels == cls
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                c=colors[i], label=class_names[i], s=12, alpha=0.6)

plt.legend(
    loc='upper right',
    fontsize=14,
    frameon=True,             # 显示边框
    facecolor='white',        # 图例背景色（纯白）
    edgecolor='lightgray',    # 边框颜色，可改为 'black' 或 '#CCCCCC'
    labelcolor='black',       # 图例文字颜色
    markerscale=1.5,
    handletextpad=0.5,
    borderpad=0.5,            # 内边距，使内容不紧贴边框
    fancybox=True,            # 圆角边框
)


plt.tight_layout()
plt.savefig('tsne_EEG.png', dpi=300)
plt.show()
