import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random

from Main_model import *

from data_input import getloader

data_dir = "../../../datasets/SEED/seed_4s/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def calculate_metrics(y_true, y_pred, average='macro'):
    """
    计算多分类任务的 Precision、Recall 和 F1-score。

    参数:
        y_true: 真实标签，Tensor
        y_pred: 预测标签，Tensor
        average:
            macro：每一类分别计算指标后直接平均
            weighted：按照每一类样本数量加权平均
            micro：整体统计 TP/FP/FN 后计算

    返回:
        precision, recall, f1
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    # 如果张量在GPU上，先移到CPU
    if y_true.is_cuda:
        y_true = y_true.cpu()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu()
    
    # 转换为NumPy数组
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    precision = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    return precision, recall, f1



    # 计算并返回F1分数
    # return f1_score(y_true, y_pred, average=average)

def get_model(subj_num=1,session_num=1):
    # model = Model2().to(device)
    Model = model(feature_dim=5, hidden_dim=5, source_num=14, classs_num=3).to(device)


    model_weight_path = f"./Pro_log/best_ACC_subj{subj_num}_session{session_num}.pth"
    # Model.load_state_dict(torch.load(model_weight_path))
    Model.load_state_dict(torch.load(model_weight_path, map_location=device))
    Model.eval()
    return Model

def acc_subj(num_subj=1,session_num=1,batch_size = 32):
    net = get_model(subj_num=num_subj, session_num=session_num)
    net = net.to(device)
    # net = model(feature_dim=5, hidden_dim=5, source_num=14, classs_num=3)
    # dataloader_target = getloader(subj_test=num_subj, session_num=session_num, batch_size=batch_size , is_drop_last=False, is_shuffle=False, is_train=False)

    tes_loader = getloader(subj_test=num_subj, session_num=session_num, batch_size=batch_size , is_drop_last=False, is_shuffle=False, is_train=False)
    tes_acc = 0
    count = 0
    with torch.no_grad():
        for step, data  in enumerate(tes_loader):
            images, labels,subj  = data
            images=images.to(device)
            labels=labels.to(device)
            subj=subj.to(device)

            # outputs,_  = net(images)
            outputs = net.predict_class(images)

            predict_y = torch.max(outputs, dim=1)[1]
            if count == 0:
                y_pred = predict_y
                y_true = labels.to(device)
            else:
                y_pred = torch.cat((y_pred, predict_y), dim=0)
                y_true = torch.cat((y_true, labels.to(device)), dim=0)
            tes_acc += (predict_y == labels).sum().item()
            count = count+1

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
        precision, recall, f1 = calculate_metrics(
            y_true,
            y_pred,
            average='macro'
        )

        print("acc:", tes_acc)
        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)

        return tes_acc, precision, recall, f1



if __name__ == '__main__':

    batch_size = 128 # 64

    best_Acc = []
    F1 = []
    Precision = []
    Recall = []

    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        best_model_acc, precision, recall, f1 = acc_subj(
            num_subj=i,
            session_num=1,
            batch_size=batch_size
        )

        best_Acc.append(best_model_acc)
        Precision.append(precision)
        Recall.append(recall)
        F1.append(f1)

    k = 0
    result_string = ''
    for i in best_Acc:
        k=k+1
        str1 = f'subject{k} acc:{i} \n'
        result_string = ''.join([result_string, str1])

    # 所有subjects的平均准确率、标准差
    mean_acc = np.mean(best_Acc)
    sta = np.std(best_Acc, ddof=1)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str]) 

    print(result_string)

    print("f1_aver:", np.mean(F1), np.std(F1, ddof=1))
    print("Precision_aver:", np.mean(Precision), np.std(Precision, ddof=1))
    print("Recall_aver:", np.mean(Recall), np.std(Recall, ddof=1))





