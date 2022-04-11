from turtle import color
import matplotlib.pyplot as plt
import os
import glob
import json
import numpy as np
def method_test_pair(folder, name ="few_shot_test_pair"):
    res = {}
    for f in glob.glob(f"{folder}/**/{name}.json", recursive=True):
        data = json.load(open(f))
        c = os.path.basename( os.path.dirname(f) )
        c = c.replace("_fold", "")
        if c in data:
            del data[c]
        print(c)
        res[c] = data
    return res

def random_test_pair(folder):
    res = {}
    for f in glob.glob(f"{folder}/**/random_pair.json", recursive=True):
        data = json.load( open(f) )
        #print(f.split("/")[3].split("_"))
        folder = f.split("/")[3].split("_")[3]
        print(folder)
        res[folder] = data
    return res

def method_test_pair_no(folder):
    res = {}
    for f in glob.glob(f"{folder}/**/few_shot_test_pair.json", recursive=True):
        data = json.load( open(f) )
        #print(f.split("/")[2].split("_"))
        folder = f.split("/")[2].split("_")[3]
        print(folder)
        res[folder] = data
    return res


def plot_performance(res1_all, res2_all, p, ks, score, t1, t2, output):
    
    id={}
    x = []
    y = []
    if score == "f1":
        i1 = 3
        i2 = 4
    if score =="recall":
        i1 = 2
        i2 = 3
    if score =="precision":
        i1 = 1
        i2 = 2
    first = True
    plt.figure(figsize=(12, 12))
    for k in ks: 
        if p not in k:
            continue
        if k not in res2_all:
            continue
        res1=res1_all[k]
        res2=res2_all[k]
        keys = set(list(res1.keys())).intersection(set(list(res2.keys())))
        keys = list(keys)
        for c in keys:
            if c not in id:
                id[c]=len(id)
        if first:
            plt.scatter( [ id[i] for i in keys ],[ res1[k]["classification"][i1] for k in keys ], label="few show",marker=".",color="b" ,alpha=0.5)
            plt.scatter( [ id[i] for i in keys ],[ res2[k][i2] for k in keys ],  label="supervised", marker="o",color="g", alpha=0.5)
            first = False
        else:
            plt.scatter( [ id[i] for i in keys ],[ res1[k]["classification"][i1] for k in keys ],marker=".",color="b" ,alpha=0.5)
            plt.scatter( [ id[i] for i in keys ],[ res2[k][i2] for k in keys ], marker="o",color="g", alpha=0.5)
        y.extend( [ res2[k][i2] for k in keys ] )
        x.extend([ res1[k]["classification"][i1] for k in keys ] )

    plt.xlabel(f"Commit ID")
    plt.ylabel(f"{score}")
    plt.legend()
    plt.savefig(f'{output}/{t1}_{t2}_{p}_{score}_total.png')
    plt.title(f'{p}, {score}')
   # plt.show()
    diff = np.asarray(x) - np.asarray(y)
    larger = (diff>0).sum()
    smaler = (diff<0).sum()
    plt.figure(figsize=(12, 12))
    # the histogram of the data
    n, bins, patches = plt.hist(diff, 50, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Difference')
    plt.ylabel('Probability')
    plt.title(f'Diff {p}, {score}, L: {larger}, S: {smaler}')
    plt.grid(True)
    plt.savefig(f'{output}/{t1}_{t2}_{p}_{score}_diff.png')
  #  plt.show()

def plot_performance_random(res1_all, res2_all, p, ks, score, t1, t2, output):
    
    id={}
    x = []
    y = []
    if score == "f1":
        i1 = 3
      
    if score =="recall":
        i1 = 2
      
    if score =="precision":
        i1 = 1
      
    first = True
    plt.figure(figsize=(12, 12))
    for k in ks: 
        if p not in k:
            continue
        if k not in res2_all:
            continue
        res1=res1_all[k]
        res2=res2_all[k]
        keys = set(list(res1.keys())).intersection(set(list(res2.keys())))
        keys = list(keys)
        for c in keys:
            if c not in id:
                id[c]=len(id)
        if first:
            plt.scatter( [ id[i] for i in keys ],[ res1[k]["classification"][i1] for k in keys ], label="few show",marker=".",color="b" ,alpha=0.5)
            plt.scatter( [ id[i] for i in keys ],[ res2[k]["classification"][i1] for k in keys ],  label="supervised", marker="o",color="g", alpha=0.5)
            first = False
        else:
            plt.scatter( [ id[i] for i in keys ],[ res1[k]["classification"][i1] for k in keys ],marker=".",color="b" ,alpha=0.5)
            plt.scatter( [ id[i] for i in keys ],[ res2[k]["classification"][i1]  for k in keys ], marker="o",color="g", alpha=0.5)
        y.extend( [ res2[k]["classification"][i1] for k in keys ] )
        x.extend([ res1[k]["classification"][i1] for k in keys ] )

    plt.xlabel(f"Commit ID")
    plt.ylabel(f"{score}")
    plt.legend()
    plt.savefig(f'{output}/{t1}_{t2}_{p}_{score}_total.png')
    plt.title(f'{p}, {score}')
  #  plt.show()
    diff = np.asarray(x) - np.asarray(y)
    larger = (diff>0).sum()
    smaler = (diff<0).sum()
    plt.figure(figsize=(12, 12))
    # the histogram of the data
    n, bins, patches = plt.hist(diff, 50, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Difference')
    plt.ylabel('Probability')
    plt.title(f'Diff {p}, {score}, L: {larger}, S: {smaler}')
    plt.grid(True)
    plt.savefig(f'{output}/{t1}_{t2}_{p}_{score}_diff.png')
   # plt.show()

def plot_kill_random():
    # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_killed_fine_tune_yes")
    res2 = random_test_pair("results/random_prior_killed/yes")
    for score in ["f1", "recall", "precision"]:
        output = f"fig/killed_few_shot_vs_random/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance_random(res1, {k:res2["csv"] for k in res2["csv"]}, "csv",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["io"] for k in res2["io"]}, "io",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["text"] for k in res2["text"]}, "text",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed",output)
        plot_performance_random(res1, {k:res2["lang"] for k in res2["lang"]}, "lang",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["collections"] for k in res2["collections"]}, "collections",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
    
    # subsuming killed mutants, few_shot_learing VS Supervirsed
    # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_killed_subsuming_fine_tune_yes")
    res2 = random_test_pair("results/random_prior_killed_subsuming/yes")
    
    for score in ["f1", "recall", "precision"]:
        output = f"fig/killed_subsuming_few_shot_vs_random/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance_random(res1, {k:res2["text"] for k in res2["text"]}, "text",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["csv"] for k in res2["csv"]}, "csv",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["io"] for k in res2["io"]}, "io",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["lang"] for k in res2["lang"]}, "lang",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["collections"] for k in res2["collections"]}, "collections",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)


def plot_relevance_random():
    # relevance mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_relevance_fine_tune_yes")
    res2 = random_test_pair("results/random_prior_relevance/yes")
    for score in ["f1", "recall", "precision"]:
        output = f"fig/relevance_few_shot_vs_random/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance_random(res1, {k:res2["csv"] for k in res2["csv"]}, "csv",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["io"] for k in res2["io"]}, "io",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["text"] for k in res2["text"]}, "text",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed",output)
        plot_performance_random(res1, {k:res2["lang"] for k in res2["lang"]}, "lang",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["collections"] for k in res2["collections"]}, "collections",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
    
    # subsuming relevance mutants, few_shot_learing VS Supervirsed
    # relevance mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_relevance_subsuming_fine_tune_yes")
    res2 = random_test_pair("results/random_prior_subsuming/yes")
    
    for score in ["f1", "recall", "precision"]:
        output = f"fig/relevance_subsuming_few_shot_vs_random/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance_random(res1, {k:res2["text"] for k in res2["text"]}, "text",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["csv"] for k in res2["csv"]}, "csv",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["io"] for k in res2["io"]}, "io",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["lang"] for k in res2["lang"]}, "lang",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance_random(res1, {k:res2["collections"] for k in res2["collections"]}, "collections",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)

def plot_kill():
    # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_killed_fine_tune_yes")
    res2 = method_test_pair("results/supervised_killed_transferweights/CE", name="test")
    
    for score in ["f1", "recall", "precision"]:
        os.makedirs(f"fig/killed_few_shot_vs_supervised/{score}", exist_ok=True)
        plot_performance(res1, res2, "csv",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", f"fig/killed_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "io",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", f"fig/killed_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "text",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", f"fig/killed_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "lang",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", f"fig/killed_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "collections",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", f"fig/killed_few_shot_vs_supervised/{score}")
    
    # subsuming killed mutants, few_shot_learing VS Supervirsed
    # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_killed_subsuming_fine_tune_yes")
    res2 = method_test_pair("results/supervised_killed_subsuming_transferweights/CE", name="test")
    
    for score in ["f1", "recall", "precision"]:
        output = f"fig/killed_subsuming_few_shot_vs_supervised/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance(res1, res2, "text",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance(res1, res2, "csv",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance(res1, res2, "io",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance(res1, res2, "lang",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)
        plot_performance(res1, res2, "collections",list(res1.keys()),score, "Few_Shot_Killed", "Supervirsed", output)

def plot_relevance():
     # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_relevance_fine_tune_yes")
    res2 = method_test_pair("results/supervised_relevance_transferweights/CE", name="test")
    
    for score in ["f1", "recall", "precision"]:
        os.makedirs(f"fig/relevance_few_shot_vs_supervised/{score}", exist_ok=True)
        plot_performance(res1, res2, "csv",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", f"fig/relevance_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "io",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", f"fig/relevance_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "text",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", f"fig/relevance_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "lang",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", f"fig/relevance_few_shot_vs_supervised/{score}")
        plot_performance(res1, res2, "collections",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", f"fig/relevance_few_shot_vs_supervised/{score}")
    
    # subsuming killed mutants, few_shot_learing VS Supervirsed
    # killed mutants, few_shot_learing VS Supervirsed
    res1 = method_test_pair("results/few_shot_relevance_subsuming_fine_tune_yes")
    res2 = method_test_pair("results/supervised_relevance_subsuming_transferweights/CE", name="test")
    
    for score in ["f1", "recall", "precision"]:
        output = f"fig/relevance_subsuming_few_shot_vs_supervised/{score}"
        os.makedirs(output, exist_ok=True)
        plot_performance(res1, res2, "text",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance(res1, res2, "csv",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance(res1, res2, "io",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance(res1, res2, "lang",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)
        plot_performance(res1, res2, "collections",list(res1.keys()),score, "Few_Shot_relevance", "Supervirsed", output)

if __name__ == "__main__":
    plot_relevance_random()
    plot_kill_random()
    plot_kill()
    plot_relevance()



