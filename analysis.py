import matplotlib.pyplot as plt
import os
import glob
import json
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


def plot_performance(res1, res2, keys, t1, t2):
    plt.figure(figsize=(12, 12))
    plt.scatter([ res1[k]["pr_area"] for k in keys ], [ i for i in range(len(keys)) ], label=f"{t1}  PR-Curve_Area", marker="v", alpha=0.5)
    plt.scatter([ res2[k]["pr_area"] for k in keys ], [ i for i in range(len(keys)) ], label=f"{t2}  PR-Curve_Area", marker="v", alpha=0.5)
    plt.xlabel(f"Commit ID")
    plt.ylabel(f"PR-Curve_Area")
    plt.legend()
   # plt.savefig(f'{t1}_{t2}_distill_PR-Curve_Area.png')
    plt.show()

if __name__ == "__main__":
    res1 = method_test_pair("results/few_shot_killed_fine_tune_yes")
    res2 = method_test_pair("results/supervised_killed_transferweights/CE", name="test")
    res3 = method_test_pair("results/supervised_killed_transferweights/SCL", name="test")
    res4 = random_test_pair("results/random_prior/no")
    res5 = method_test_pair_no("results/few_shot_killed_fine_tune_no")
    for k in res1:
        plot_performance(res1[k], res2[k], list(res1[k].keys()), "Few_Shot_Killed", "Supervirsed")