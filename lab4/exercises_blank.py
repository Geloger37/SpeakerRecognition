# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    assert len(all_scores) == len(all_labels)
    tar_scores = [all_scores[it] for it, el in enumerate(all_labels) if el == 1]
    imp_scores = [all_scores[it] for it, el in enumerate(all_labels) if el == 0]
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    indexes = np.argsort(all_scores)
    all_scores_sort = all_scores[indexes]
    ground_truth_sort = all_labels[indexes] == 1
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)

    LLR = np.log(tar_gauss_pdf / imp_gauss_pdf)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    P_Htar  = 1/2

    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)

    idx = np.argmin(np.abs(fnr_thr - fnr)) 

    fpr = fpr_thr[idx]
    thr = LLR[idx]
    ###########################################################

    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    
    len_thr = len(LLR)
    AC_err   = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        
        solution = LLR > LLR[idx]                    

        tr = (solution == ground_truth_sort)
        err = (solution != ground_truth_sort)

        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores)
        tpr_thr[idx] = np.sum(tr [ ground_truth_sort])/len(tar_scores)
        tnr_thr[idx] = np.sum(tr [~ground_truth_sort])/len(imp_scores)
        
        AC_err[idx] = C00 * tpr_thr[idx] * P_Htar + C10 * fnr_thr[idx] * P_Htar + C01 * fpr_thr[idx] * (1 - P_Htar) + C11 * tnr_thr[idx] * (1 - P_Htar)


    AC_err_idx = np.argmin(AC_err)
    AC_err_min = AC_err.min()

    thr, fnr, fpr, AC = LLR[AC_err_idx], fnr_thr[AC_err_idx], fpr_thr[AC_err_idx], AC_err_min

    ###########################################################

    return thr, fnr, fpr, AC

from time import time
from numba import njit, prange

@njit(parallel=True)
def bayes(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):

    len_thr = len(LLR)
    # len_thr = 1000
    AC_err  = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)

    # LLR_trim = np.linspace(-1.0, 0.5, 1000)
    
    for idx in prange(len_thr):
        
        # solution = LLR > LLR_trim[idx]
        solution = LLR > LLR[idx]                    

        tr = (solution == ground_truth_sort)
        err = (solution != ground_truth_sort)

        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores)
        tpr_thr[idx] = np.sum(tr [ ground_truth_sort])/len(tar_scores)
        tnr_thr[idx] = np.sum(tr [~ground_truth_sort])/len(imp_scores)
        
        AC_err[idx] = C00 * tpr_thr[idx] * P_Htar + C10 * fnr_thr[idx] * P_Htar + C01 * fpr_thr[idx] * (1 - P_Htar) + C11 * tnr_thr[idx] * (1 - P_Htar)


    AC_err_idx = np.argmin(AC_err)
    AC_err_min = AC_err.min()

    # thr, fnr, fpr, AC = LLR_trim[AC_err_idx], fnr_thr[AC_err_idx], fpr_thr[AC_err_idx], AC_err_min
    
    thr, fnr, fpr, AC = LLR[AC_err_idx], fnr_thr[AC_err_idx], fpr_thr[AC_err_idx], AC_err_min
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = -10**10
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code

    for idx in range(len(P_Htar_thr)):

        # st = time()
        P_H0 = P_Htar_thr[idx]

        llr_thr_val, fnr_thr_val, fpr_thr_val, AC_val = bayes(ground_truth_sort, LLR, tar_scores, imp_scores, P_H0, C00, C10, C01, C11)

        if AC_val > AC:
            thr, fnr, fpr, AC, P_Htar = llr_thr_val, fnr_thr_val, fpr_thr_val, AC_val, P_H0
        
        # print(f'Execution time: {time() - st}')
        
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar