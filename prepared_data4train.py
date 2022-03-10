from matplotlib import collections
#from utils.mutantsdataset_siamese import MutantsDataset
from utils.mutantsdataset import MutantRelevanceDataset, MutantKilledDataset
import argparse
import json
from torch_geometric.data import DataLoader
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import collections
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    for p in projects:
        #set up dataset dataset_path
        print(p)
        if args.task == "relevance":
            dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p, probability=0)
        else:
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p)
        dataset_list[p] = dataset_inmemory
    return dataset_list


def projects_dict(args):
    projects = collections.defaultdict(list)
    name=[]  
    for p in args.projects:
        for pf in glob.glob(f"{args.dataset_path}/{p}*"):
            projects[p].append(os.path.basename(pf))
            name.append( os.path.basename(pf) )
   
    return projects, name


def prepparedata(args):
    projects, namelist = projects_dict(args)
    dataset_list = fecth_datalist(args, namelist)   
    with open( f"relevance_sr.txt", "w") as fr, open(f"relevance_fixed_b.txt", "w") as fb, open(f"relevance_fixed_m.txt", "w") as fm:
        fr.write("test,val,train,name \n")
        fb.write("test,val,train,name \n")
        fm.write("test,val,train,name \n")
        for p in dataset_list:
            dataset = dataset_list[p]
            test, val, train = dataset.splitting_ratio()
            fr.write(f"{test}, {val}, {train}, {p} \n")
            if args.task == "relevance":
                test, val, train = dataset.splitted_fixed(dataset.relevance_mutant_binary_labels, fixed_train=10)
                fb.write(f"{test}, {val}, {train}, {p} \n")
                test, val, train = dataset.splitted_fixed(dataset.relevance_mutant_multiple_labels, fixed_train=10)
                fm.write(f"{test}, {val}, {train}, {p} \n")
            else:
                test, val, train = dataset.splitted_fixed(dataset.kill_mutant_binary_labels, fixed_train=10)
                fb.write(f"{test}, {val}, {train}, {p} \n")
                test, val, train = dataset.splitted_fixed(dataset.kill_mutant_multiple_labels, fixed_train=10)
                fm.write(f"{test}, {val}, {train}, {p} \n")
  

   


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument("--dataset_path", type=str, default="dataset/pittest")
    parser.add_argument("--dataset", type=str, default="DV_PDG")
    parser.add_argument("--task", type=str, default="relevance")
    parser.add_argument("--projects", nargs="+", default=["collections", "csv", "io", "text", "lang"])
    parser.add_argument('--seed', type = int, default =1234)
    args = parser.parse_args( )
    set_seed(args)
    prepparedata(args)

    
if __name__ == "__main__":
    main()

