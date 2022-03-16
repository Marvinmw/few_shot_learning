import torch
 # top k performance
from utils.metrics import mean_reciprocal_ranks, average_precision, precision_at_k
from utils.tools import performance
from sklearn.metrics import average_precision_score,roc_auc_score, auc, precision_recall_curve
import numpy as np


def threshold_sigmoid(similarity, th=0.5):
    threashold = np.zeros_like( similarity )
    threashold.fill(th)
    return ( similarity > threashold ).reshape(-1, 1)

def ranking_performance(ground_truth_np, score_pre):
    # ROC-Area
    roc_score = roc_auc_score( ground_truth_np, score_pre  )
    # Compute Precision-Recall and plot curve
    precision, recall, thresholds = precision_recall_curve(ground_truth_np, score_pre)
    pr_area = auc(recall, precision)   

    # th = 0.5
    predicted_label = threshold_sigmoid( score_pre ).astype(np.int)
    acc, pre, re, f = performance(ground_truth_np.astype(np.int), predicted_label, average="binary")

    # top k performance
    from utils.metrics import mean_reciprocal_ranks, average_precision, precision_at_k
    from sklearn.metrics import average_precision_score
    #mrr = mean_reciprocal_ranks( ground_truth_np, score_pre )
    map = average_precision_score( ground_truth_np, score_pre )
    
    # top k AP
    top_k = [ 1, 5, 10, 20, 30]
    precision_top_k = []
    for k in top_k:
        if ground_truth_np.size < k:
            break
        #print( precision_at_k( ground_truth_np, max_score, k  ) )
        precision_top_k.append( precision_at_k( ground_truth_np, score_pre, k  ) )
    
    #logger.info( f"{testdataset.project},roc_auc_score {roc_score} PR-Area {pr_area}, mean_reciprocal_ranks {mrr}, mean_avg_precision {map}, precision@k, { precision_top_k }, predciton {acc, pre, re, f }" )
    #if "lang_25" == testdataset.project:
    #    logger.info( f"range {np.max(score), np.min(score)}" )

    res = { "map":map, "precision@k":precision_top_k, "roc_auc_score": roc_score, "pr_area":pr_area,"classification":[ acc, pre, re, f] }
    return res