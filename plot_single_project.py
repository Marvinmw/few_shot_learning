import json
import glob
import collections
import pandas as pd
import matplotlib.pyplot as plt
acc_yes = []
pre_yes = []
recall_yes = []
f1_yes = []

acc_no = []
pre_no = []
recall_no = []
f1_no = []

def read_performance(project, datapath="single_projects"):
    pdir = f"{datapath}/mutants_class_contrastive_2_{project}_no"
    sum_data = collections.defaultdict(list)
    sum_stat = collections.defaultdict(list)
    for pdir in [f"{datapath}/mutants_class_contrastive_2_{project}_yes", f"{datapath}/mutants_class_contrastive_2_{project}_no"]:
        for f in glob.glob(f"{pdir}/**/gat_test_*"):
            data = json.load(open(f"{f}/performance.json", "r"))["test"]
            stat = json.load(open(f"{f}/stat.json", "r"))
            train_size = sum(stat["train"].values())+sum(stat["val"].values())
            test_size = sum(stat["test"].values())
            for t in ["train", "test", "val"]:
                if "0" not in stat[t]:
                    stat["test"]["0"] = 0 
            pversion = list( data.keys() )[0]
            [loss, acc, precision, recall, f1] = data[pversion][:5]
            sum_data[pversion].extend( [pversion, loss, acc, precision, recall, f1, train_size, test_size, (stat["train"]["0"]+ stat["val"]["0"])/train_size,
            stat["test"]["0"]/test_size ] )
      

    df = pd.DataFrame.from_dict( sum_data, orient="index", columns=["name_yes","loss_yes", "acc_yes", "precision_yes", "recall_yes", "f1_yes", 
            "train_size_yes", "test_size_yes", "train_live_yes", "test_live_yes", "name_no", "loss_no", "acc_no", "precision_no", "recall_no", "f1_no", 
            "train_size_no", "test_size_no", "train_live_no", "test_live_no"] )
    acc_yes.extend(df["acc_yes"].tolist())
    pre_yes.extend(df["precision_yes"].tolist())
    recall_yes.extend(df["recall_yes"].tolist())
    f1_yes.extend(df["f1_yes"].tolist())

    acc_no.extend(df["acc_no"].tolist())
    pre_no.extend(df["precision_no"].tolist())
    recall_no.extend(df["recall_no"].tolist())
    f1_no.extend(df["f1_no"].tolist())

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    df.plot.bar( "name_yes", ["acc_yes", "acc_no"], ax=axes[0][0])
    df.plot.bar("name_yes", ["precision_yes", "precision_no"],  ax=axes[0][1])
    df.plot.bar("name_yes", ["recall_yes", "recall_no"],  ax=axes[1][0])
    df.plot.bar("name_yes", ["f1_yes", "f1_no"],  ax=axes[1][1])
    plt.suptitle(project)
    plt.savefig(f"{project}.pdf")
    plt.savefig(f"{project}.jpg")
    plt.show()



#read_performance("Gson")
#read_performance("JacksonCore")
#read_performance("JacksonXml")
#read_performance("Lang")
#read_performance("Math")
#read_performance("Mockito")
#read_performance("Time")
read_performance("Cli")
read_performance("Csv")
read_performance("Compress")
read_performance("Chart")
read_performance("Jsoup")
read_performance("JxPath")

from scipy.stats import mannwhitneyu
U1, p = mannwhitneyu(acc_yes, acc_no, alternative="greater")
print(p)
U1, p = mannwhitneyu(pre_yes, pre_no, alternative="greater")
print(p)
U1, p = mannwhitneyu(recall_yes, recall_no, alternative="greater")
print(p)
U1, p = mannwhitneyu(f1_yes, f1_no, alternative="greater")
print(p)