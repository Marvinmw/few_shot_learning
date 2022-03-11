import matplotlib.pyplot as plt
import json
import glob 
import os
def scatter_plot(data_folder, name):
    data1 = []
    data2 = []
    for test_file in glob.glob(f"{data_folder}/**/test.json", recursive=True):
        print(test_file)
        folder = os.path.dirname(test_file)
        p_file = os.path.join(folder, "performance.json")
        test_data = json.load( open(test_file) )
        train_performance = json.load(open( p_file ))
        val_f1 = train_performance["val_f1"]
        for pid in test_data:
            test_f1 = test_data[pid][5]["eval_f1"]
            distill_f1 = test_data[pid][6]["eval_f1"]
            data1.append((val_f1, test_f1))
            data2.append((val_f1, distill_f1))

    plt.figure(figsize=(12, 12))
    plt.scatter([ d[0] for d in data1 ], [ d[1] for d in data1 ], label="Test  F1", marker="v", alpha=0.5)
    plt.scatter([ d[0] for d in data2 ], [ d[1] for d in data2 ], label="Test Distill F1", marker="h",alpha=0.5)
    plt.xlabel("Val F1")
    plt.ylabel("Test F1")
    plt.legend()
    plt.savefig(f'{name}.png')

def get_f1(datafolder):
    data1 = []
    data2 = []
    for test_file in glob.glob(f"{datafolder}/**/test.json", recursive=True):
        print(test_file)
        folder = os.path.dirname(test_file)
        p_file = os.path.join(folder, "performance.json")
        test_data = json.load( open(test_file) )
        train_performance = json.load(open( p_file ))
        val_f1 = train_performance["val_f1"]
        for pid in test_data:
            test_f1 = test_data[pid][5]["eval_f1"]
            distill_f1 = test_data[pid][6]["eval_f1"]
            data1.append((val_f1, test_f1))
            data2.append((val_f1, distill_f1))
    return data1, data2

def scatter_plot_two_loss(data_folder,l1,l2, name):
    data_folder1 = f"{data_folder}/mutants_relevance_2_loss_{l1}_rm_{name}"
    data_foler2 = f"{data_folder}/mutants_relevance_2_loss_{l2}_rm_{name}"
    data1, data2 = get_f1(data_folder1)
    data3, data4 = get_f1(data_foler2)

    plt.figure(figsize=(12, 12))
    plt.scatter([ d[0] for d in data1 ], [ d[1] for d in data1 ], label=f"{l1}  F1", marker="v", alpha=0.5)
    plt.scatter([ d[0] for d in data3 ], [ d[1] for d in data3 ], label=f"{l2} F1", marker="h",alpha=0.5)
    plt.xlabel(f"val F1")
    plt.ylabel(f"{l2} F1")
    plt.legend()
    plt.savefig(f'{l2}_{l1}_{name}.png')


    plt.figure(figsize=(12, 12))
    plt.scatter([ d[0] for d in data1 ], [ d[1] for d in data1 ], label=f"{l1}  F1", marker="v", alpha=0.5)
    plt.scatter([ d[0] for d in data3 ], [ d[1] for d in data3 ], label=f"{l2} F1", marker="h",alpha=0.5)
    plt.xlabel(f"val F1")
    plt.ylabel(f"{l2} F1")
    plt.legend()
    plt.savefig(f'{l2}_{l1}_distill_{name}.png')


import sys
if len(sys.argv) == 3:
    scatter_plot(sys.argv[1], sys.argv[2])

if len(sys.argv) == 5:
    scatter_plot_two_loss(sys.argv[1], sys.argv[2],sys.argv[3], sys.argv[4])