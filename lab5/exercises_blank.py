# Exercises in order to perform laboratory work

# Import of modules
import numpy as np
from math import sqrt
import itertools
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from sklearn.metrics.pairwise import cosine_similarity
from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    # Function to get target and impostors scores based on the labels

    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):

        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])

        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores

def plot_histograms_2sets(all_scores_1, all_labels_1,
                          all_scores_2, all_labels_2,
                          names=['in-domain', 'out-of-domain']):
    # Function to show target/impostor histograms and compute EER to out-of-domain and in-domain datasets
    
    # Get target and impostors scores
    tar_scores_1, imp_scores_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    tar_scores_2, imp_scores_2 = get_tar_imp_scores(all_scores_2, all_labels_2)

    # Plot histograms for target and impostor scores
    min_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).min()
    max_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).max()

    hist(tar_scores_1, int(sqrt(len(tar_scores_1))), histtype='step', color='green',
         range=(min_scores, max_scores))
    hist(imp_scores_1, int(sqrt(len(imp_scores_1))), histtype='step', color='red',
         range=(min_scores, max_scores))
    hist(tar_scores_2, int(sqrt(len(tar_scores_2))), histtype='step', color='blue',
         range=(min_scores, max_scores))
    hist(imp_scores_2, int(sqrt(len(imp_scores_2))), histtype='step', color='cyan',
         range=(min_scores, max_scores))
    xlabel('$s$');
    ylabel('$\widehat{W}(s|H_0)$, $\widehat{W}(s|H_1)$');
    title('VoxCeleb1-O (cleaned), histograms');
    legend(list('{}_{}'.format(a[0], a[1]) for a in itertools.product(names, ['tar', 'imp'])))
    grid()
    show()

    # Compute equal error rate
    EER_1, thresh_EER_1 = get_eer(tar_scores_1, imp_scores_1)
    EER_2, thresh_EER_2 = get_eer(tar_scores_2, imp_scores_2)

    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[0], EER_1, thresh_EER_1))
    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[1], EER_2, thresh_EER_2))

def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    """
    Function to perform s-normalization for scores with the snorm_data
    :param test_data: test embeddings
    :param lines: test protocol
    :param adapt_data: data for s-norm (s-norm embeddings)
    :param N_s: top N impostors scrores for s-normalization
    :param eps: epsilon for std
    :return: snorm_scores - s-normalized scores
    """
    
    scores_adapted = []
    all_labels = []
    all_trials = []

    # Prepare lists of unique wavs from protocols
    enroll_list = list(set(list([x.strip().split()[1] for x in lines])))
    test_list   = list(set(list([x.strip().split()[2] for x in lines])))
    adapt_list  = list(adapt_data.keys())

    # Prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
    E = []
    for id, enr in enumerate(enroll_list):
        E.append(test_data[enr].squeeze(0).numpy())
    E = np.array(E)

    # Prepare tests: save test embds in ndarray [num_wavs x emb_size]
    T = []
    for id, tst in enumerate(test_list):
        T.append(test_data[tst].squeeze(0).numpy())
    T = np.array(T)

    # Prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
    A = []
    for id, a in enumerate(adapt_list):
        A.append(adapt_data[a].squeeze(0).numpy())
    A = np.array(A)
    
    ###########################################################
    # Here is your code
    sorted_scores_e = (-np.sort(-cosine_similarity(E, A), axis=1))[:, :N_s]
    sorted_scores_t = (-np.sort(-cosine_similarity(T, A), axis=1))[:, :N_s]

    print(f'{sorted_scores_e.shape}')
    print(f'{sorted_scores_t.shape}')

    mean_e = sorted_scores_e.mean(axis=1)
    mean_t = sorted_scores_t.mean(axis=1)
    std_e = sorted_scores_e.std(axis=1) + eps
    std_t = sorted_scores_t.std(axis=1) + eps

    print(f'{mean_e.shape}')
    print(f'{mean_t.shape}')
    print(f'{std_e.shape}')
    print(f'{std_t.shape}')

    for line in tqdm.tqdm(lines):
        label, enr, tst = line.strip().split()
        all_labels.append(int(label))
        all_trials.append(" ".join([enr, tst]))
        _E = test_data[enr].squeeze(0).numpy().reshape(1, -1)
        _T = test_data[tst].squeeze(0).numpy().reshape(1, -1)
        s = cosine_similarity(_E, _T)[0][0]

        s_norm = (s - mean_e[enroll_list.index(enr)])/(std_e[[enroll_list.index(enr)]]) + (s - mean_t[test_list.index(tst)])/(std_t[test_list.index(tst)])

        scores_adapted.append(s_norm[0])
        
    ###########################################################

    return scores_adapted, all_labels, all_trials

class CalibrationDataset(Dataset):

    def __init__(self, target_scores, impostor_scores):
        super().__init__()

        self.target_scores   = target_scores
        self.impostor_scores = impostor_scores
        self.L_tar = len(target_scores)
        self.L_imp = len(impostor_scores)

    def __len__(self):
        
        return 1

    def __getitem__(self, idx):

        return self.target_scores, self.impostor_scores

class LinearCalibrationModel(torch.nn.Module):
    # Building of the full model for constructing the extractor of features
    
    def __init__(self):
        super().__init__()
        
        self.calib_params = nn.Linear(1, 1)

    def forward(self, x):
        
        ###########################################################
        # Here is your code
        calib_x = self.calib_params(x)
            
        ###########################################################

        return calib_x
    
class CalibrationLoss(nn.Module):

    def __init__(self, ptar=0.01):
        '''
        Calibration loss. Code is based on https://github.com/alumae/sv_score_calibration/blob/master/calibrate_scores.py
        :param ptar: probability of target hypothesis
        '''
        
        super().__init__()
        
        self.ptar  = ptar
        self.alpha = np.log(ptar/(1 - ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            # Function to compute -log(sigmoid(log_odds))
            
            return torch.log1p(torch.exp(-lodds))
        
        loss_value = 0
        
        ###########################################################
        # Here is your code
        loss_value = self.ptar * torch.mean(negative_log_sigmoid(target_llrs - self.alpha)) + (1 - self.ptar) * torch.mean(negative_log_sigmoid(-nontarget_llrs + self.alpha))

        ###########################################################

        return loss_value

def train_calibration(train_loader, model, criterion, optimizer, scheduler, num_epochs, verbose=False):
    # Function to train calibration model
    
    model.train()
    
    for epoch in range(0, num_epochs):
        
        for batch_idx, batch_data in enumerate(train_loader):
            tar_sc = batch_data[0]
            imp_sc = batch_data[1]
            
            ###########################################################
            # Here is your code
            optimizer.zero_grad()

            # raise Exception(f'{tar_sc.shape=}\n{imp_sc.shape=}')

            new_nontarget_llrs = model(imp_sc.reshape(-1, 1))
            new_target_llrs = model(tar_sc.reshape(-1, 1))

            loss = criterion(new_target_llrs, new_nontarget_llrs)

            loss.backward()
            optimizer.step()

            
            ###########################################################
                            
        lr_value = optimizer.param_groups[0]['lr']

        if verbose:
            print("Epoch {:1.0f}, LR {:f} Loss {:f}".format(epoch, lr_value, loss.item()))
                
        scheduler[0].step()
    
    return