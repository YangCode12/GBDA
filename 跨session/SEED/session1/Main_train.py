import torch
import numpy as np
import torch.optim as optim
import os, random
from Main_model import *
from torch.utils.data import DataLoader
from itertools import cycle
from metrics import *
import glob
from data_input import getloader, SEED_dataset_train_with_GB


def seed_torch(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

def cre_prolog():
    current_script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script_path)
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log')
    if not os.path.exists(pro_log_dir_path):
        os.makedirs(pro_log_dir_path)
        print(f"Directory '{pro_log_dir_path}' created.")
    return pro_log_dir_path


data_dir = "../../../datasets/SEED/seed_4s/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
exp_dir = cre_prolog()
log_file = f"{exp_dir}/log.txt"


def compute_gb_weights(centers, targets, radii, alpha=0.5, tau=1.0, q=5):
    c_flat = centers.view(centers.size(0), -1)
    t_flat = targets.view(targets.size(0), -1)
    c_norm = c_flat / c_flat.norm(dim=1, keepdim=True)
    t_norm = t_flat / t_flat.norm(dim=1, keepdim=True)
    dist_mat = torch.cdist(c_norm, t_norm) ** 2
    r_adjusted = (radii * alpha).view(-1, 1)
    delta = torch.clamp(dist_mat - r_adjusted, min=0)
    k = min(q, targets.size(0))
    topk_vals, _ = torch.topk(delta, k, dim=1, largest=False)
    D_t = topk_vals.mean(dim=1)
    omega = torch.exp(-D_t / tau)
    return omega.detach()


def train_session(num_subj=1, session_num=1, batch_size=32, lr=0.001, epochs=240):

    net = model(classs_num=3).to(device)

    ds = SEED_dataset_train_with_GB(subj_test=num_subj, session_num=session_num)
    dataloader_source = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    dataloader_target = getloader(subj_test=num_subj, session_num=session_num, batch_size=batch_size, is_drop_last=False, is_shuffle=False, is_train=False)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_acc = 0
    best_epoch = 0
    len_dataloader = min(len(dataloader_source), len(dataloader_target))

    for epoch in range(epochs):
        tra_mmdloss = 0.0
        tra_classloss = 0.0

        data_source_iter = iter(dataloader_source)
        data_target_iter = cycle(dataloader_target)

        for i in range(len_dataloader):
            data_samples_s, labels_s, radii_s, domain_s = next(data_source_iter)
            data_samples_t, labels_t, domain_t = next(data_target_iter)
            data_samples_s, labels_s, radii_s = data_samples_s.to(device), labels_s.to(device), radii_s.to(device)
            data_samples_t, labels_t = data_samples_t.to(device), labels_t.to(device)

            net.train()

            gb_weights = compute_gb_weights(
                data_samples_s, data_samples_t, radii_s,
                alpha=0.5, tau=1.0, q=5
            )

            mmdloss, classloss, logits = net(data_samples_s, data_samples_t, labels_s, gb_weights=gb_weights)

            tra_mmdloss += mmdloss.item()
            tra_classloss += classloss.item()

            loss = classloss + 0.1 * mmdloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data_t, label_t, subj_t in dataloader_target:
                    data_t, label_t = data_t.to(device), label_t.to(device)

                    logits = net(data_t)
                    pred = logits.argmax(dim=1)

                    correct += (pred == label_t).sum().item()
                    total += label_t.size(0)

            acc_test = correct / total
            if acc_test > best_acc:
                old_model_pattern = os.path.join(exp_dir, f"best_ACC_subj{num_subj}_session{session_num}*.pth")
                for old_file in glob.glob(old_model_pattern):
                    try:
                        os.remove(old_file)
                    except Exception as e:
                        print(f"Error removing {old_file}: {e}")

                best_acc = acc_test
                best_epoch = epoch + 1
                best_model_path = os.path.join(exp_dir, f"best_ACC_subj{num_subj}_session{session_num}.pth")
                torch.save(net.state_dict(), best_model_path)

            print(f'\n Epoch {epoch} tes_acc={acc_test:.6f}, best_acc={best_acc:.6f}, best_epoch: {best_epoch} ')
            with open(log_file, 'a') as file:
                line_to_write = f'Epoch {epoch} tes_acc={acc_test:.6f}, best_acc={best_acc:.6f}, best_epoch: {best_epoch} \n'
                file.write(line_to_write)

        avg_mmdloss_train = tra_mmdloss / len_dataloader
        avg_classloss_train = tra_classloss / len_dataloader

        if best_acc >= 1.0:
            break
    return best_acc, best_epoch

if __name__ == '__main__':

    import sys
    batch_size = 128
    lr = 0.002
    epoch = 300

    best_ACC = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        with open(log_file, 'a') as file:
            file.write(f'\n---------------------------------------------------------------------start subect:{i}---------------------------------------------------------------------\n ')
        best_acc, best_epoch = train_session(num_subj=i, session_num=1, batch_size=batch_size, lr=lr, epochs=epoch)

        best_ACC.append(best_acc)

    result_lines = []
    for idx, acc in enumerate(best_ACC, 1):
        result_lines.append(f'subject{idx} ACC:{acc:.6f}\n')

    mean_acc = np.mean(best_ACC)
    std_acc  = np.std(best_ACC, ddof=0)
    result_lines.append(f'mean_ACC:{mean_acc:.6f}  std:{std_acc:.6f}')

    result_str = ''.join(result_lines)
    with open(log_file, 'a') as f:
        f.write(result_str)
    print(result_str)
