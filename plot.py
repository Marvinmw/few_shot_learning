from turtle import position
import pandas as pd
import matplotlib.pyplot as plt
import json
import collections

from torch import poisson

def plotbart(data_dir, name):
    ct_file = f"{data_dir}/single_{name}_mutants_class_contrastive_2_loss_both/context/gat/performance.json"
    ce_file = f"{data_dir}/single_{name}_mutants_class_contrastive_2_loss_CE/context/gat/performance.json"
    cts_file = f"{data_dir}/single_{name}_mutants_class_scratch_2_loss_both/context/gat/performance.json"
    ces_file = f"{data_dir}/single_{name}_mutants_class_scratch_2_loss_CE/context/gat/performance.json"
    k = {ct_file: "Siamese-pretrained", ce_file:"Supervised-pretrained", cts_file:"Siamese-scratch", ces_file:"Supervised-scratch"}
    data = collections.defaultdict(list)
    for f in [ ct_file, ce_file, cts_file, ces_file ]:
        test_performance = json.load( open(f, "r") )["test"][5]
        eval_accuracy = test_performance["eval_accuracy"]
        eval_precision = test_performance["eval_precision"]
        eval_recall = test_performance["eval_recall"]
        eval_f1 = test_performance["eval_f1"]
        data[k[f]].extend( [  eval_accuracy, eval_precision,  eval_recall, eval_f1] )
    
    df = pd.DataFrame.from_dict( data, orient="index", columns=["acc", "pre", "recall", "f1"] )
    print(df.T)
  #  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
  #  plt.figure(figsize=(12,6))
    df.T.plot.bar(color=[ "blue", "royalblue", "pink", "red"], figsize=(12,6))
    plt.suptitle(name)
    plt.legend(loc="lower right")
    # plt.savefig(f"{name}.pdf")
    plt.savefig(f"{name}_exp.jpg")
    plt.show()

def printstat(data_dir, name):
    stat_file = f"{data_dir}/single_{name}_mutants_class_contrastive_2_loss_both/context/gat/stat.json"
    stat = json.load( open(stat_file, "r") )
    train = sum(stat["train"].values())
    val = sum(stat["val"].values())
    test = sum(stat["test"].values())
    print(f"{name},  {train},  {val},  {test}")
for p in ["Closure", "JacksonCore", "JxPath", "Math", "Mockito", "Time"]:   
    printstat("exp2", p)
    plotbart("exp2", p)