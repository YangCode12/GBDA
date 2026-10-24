import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def load_predictions(path):
    # 读取保存的y_true和y_score
    checkpoint = torch.load(path)
    y_true = checkpoint['y_true']
    y_score = checkpoint['y_score']

    return y_true, y_score

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# SEED 的3个session
y_true_seed_session_1, y_score_seed_session_1 = load_predictions(f'D:/文章/hunxiao/SEED cross session_session{1}.pth')
y_true_seed_session_2, y_score_seed_session_2 = load_predictions(f'D:/文章/hunxiao/SEED cross session_session{2}.pth')
y_true_seed_session_3, y_score_seed_session_3 = load_predictions(f'D:/文章/hunxiao/SEED cross session_session{3}.pth')
# y_true_seed_session_4, y_score_seed_session_4 = load_predictions(f'./SEED_VII_session{4}.pth')
y_true_seed = torch.cat((y_true_seed_session_1, y_true_seed_session_2, y_true_seed_session_3), dim=0)
y_score_seed = torch.cat((y_score_seed_session_1, y_score_seed_session_2, y_score_seed_session_3), dim=0)
# y_prob_seed = torch.cat((y_prob_seed_session_1, y_prob_seed_session_2, y_prob_seed_session_3), dim=0)

y_true_seed = y_true_seed.cpu().numpy() 
y_score_seed = y_score_seed.cpu().numpy()
# y_prob_seed =y_prob_seed.cpu().numpy()

print('y_true shape:', y_true_seed.shape)
print('y_score shape:', y_score_seed.shape)
# print('y_prob shape:', y_prob_seed.shape)


def plot_confusion_matrix(y_true, y_score, class_names, ax, title, annot_fontsize):
    cm = confusion_matrix(y_true, y_score)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

    # 绘制混淆矩阵
    heatmap = sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar=True
    )

    # 设置右侧 colorbar 数轴字体大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)   # 这里控制右侧数轴数字大小

    ax.set_xlabel('Predicted classes (%)', fontsize=18, fontstyle='italic')
    ax.set_ylabel('True classes (%)', fontsize=18, fontstyle='italic')

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

# 创建一个计算混淆矩阵并转换为百分比的函数
# def plot_confusion_matrix(y_true, y_score, class_names, ax, title, annot_fontsize):
#     cm = confusion_matrix(y_true, y_score)
#     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比
#     sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax,
#                 annot_kws={"size": annot_fontsize})  # 设置数字字体大小
#     ax.set_xlabel('Predicted classes (%)', fontsize=18, fontstyle='italic')
#     ax.set_ylabel('True classes (%)', fontsize=18, fontstyle='italic')
#     # ax.set_title(title, fontsize=18)  # 标题设置到图片下方并调整字体大小
#     ax.tick_params(axis='x', labelsize=18)
#     ax.tick_params(axis='y', labelsize=18)


# 创建一个 2x2 的子图，用来显示四个混淆矩阵
fig, axes = plt.subplots(1, 1, figsize=(8, 7))

# 对每个混淆矩阵调用绘图函数，并调整标题
plot_confusion_matrix(y_true_seed, y_score_seed, class_names=['Sad', 'Neutral','Happy'], ax=axes, title='', annot_fontsize=24)


# 调整布局，避免重叠
plt.tight_layout()

# 保存图片到本地，设置分辨率为 300 DPI
save_path = r'D:/文章/hunxiao/confusion_matrices_SEED_cross session.png'
plt.savefig(save_path, dpi=300)

plt.show()

