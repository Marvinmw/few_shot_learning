from sklearn.dummy import DummyClassifier
import sys
# setting path
sys.path.append('../')
from utils.mutantsdataset import MutantKilledDataset, MutantRelevanceDataset, MutantTestRelevanceDataset
import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np
from utils.tools import get_logger

import collections
import random
from utils.ranking import ranking_performance

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance":
            dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p, probability=0.8 )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list

def create_dataset(args, train_projects, dataset_list):
    data = []
    for tp in train_projects:
            dataset_inmemory = dataset_list[tp] 
            dataset = dataset_inmemory.data
            data.extend( dataset )
   
    random.shuffle(data)
    train_y = [ d.by.item() for d in data ]
    train_x = [ i for i in range(len(train_y))]
    return train_y, train_x

def create_singledataset(args, train_projects, dataset_list):
    train_y = []
    for tp in train_projects:
            dataset_inmemory = dataset_list[tp] 
            dataset = dataset_inmemory.query_mutants
            train_y += [ d.label_r_binary for d in dataset ]
    train_x = [ i for i in range(len(train_y))]
    return train_y, train_x

def projects_dict(args, project_list=None):
    projects = collections.defaultdict(list)
    name=[]
    if project_list is None:
        project_list = args.projects

    if len(project_list) > 1:
        for p in project_list:
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
                projects[p].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    elif len(project_list) == 1:
        for p in project_list:
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
                n=os.path.basename(pf)
                projects[n].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    return projects, name

import glob

def fetch_testdata(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance":
            dataset_inmemory = MutantTestRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list


def test_pair(args):
    projects, namelist = projects_dict(args, args.projects)
    
    train_projects = []
    klist = list( projects.keys() )
    for pp in klist:
        train_projects.extend(projects[pp])
    orgsavedpat=args.saved_model_path

    dataset_list = fecth_datalist(args, namelist)
    args.saved_model_path = f"{orgsavedpat}"
    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    logger.info(args.saved_model_path)
    
    # learn y prior
    train_y, train_x = create_dataset(args, train_projects, dataset_list)
    stat = collections.Counter(train_y)
    json.dump(train_projects, open(os.path.join(args.saved_model_path, "train_projects.json"), "w")  )
    json.dump(stat, open(os.path.join(args.saved_model_path, "stat.json"), "w")  , indent=6)
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(train_x, train_y)
    
    # predict 
    test_projects, test_namelist = projects_dict(args, args.test_projects)
    test_dataset_list = fecth_datalist(args, test_namelist)
    sum_res = {}
    for p in test_dataset_list:
        dataset = test_dataset_list[p]
        ground_truth = [ d.by.item() for d in dataset ] 
        dummpy_X = [ i for i in range(len(ground_truth))]
        probability = dummy_clf.predict_proba(dummpy_X)[:, 1]
        res = ranking_performance(np.asarray(ground_truth),  probability)
        sum_res[p] = res
    return sum_res
    
def test_single(args):
    projects, namelist = projects_dict(args, args.projects)
    
    train_projects = []
    klist = list( projects.keys() )
    for pp in klist:
        train_projects.extend(projects[pp])
    orgsavedpat=args.saved_model_path

    dataset_list = fetch_testdata(args, namelist)
    args.saved_model_path = f"{orgsavedpat}"
    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    logger.info(args.saved_model_path)
    
    # learn y prior
    train_y, train_x = create_singledataset(args, train_projects, dataset_list)
    stat = collections.Counter(train_y)
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(train_x, train_y)

    # predict 
    test_projects, test_namelist = projects_dict(args, args.test_projects)
    test_dataset_list = fetch_testdata(args, test_namelist)
    sum_res = {}
    for p in test_dataset_list:
        dataset = test_dataset_list[p].query_mutants
        ground_truth = [ d.label_r_binary.item() for d in dataset ] 
        dummpy_X = [ i for i in range(len(ground_truth))]
        probability = dummy_clf.predict_proba(dummpy_X)[:, 1]
        res = ranking_performance(np.asarray(ground_truth),  probability)
        sum_res[p] = res
    return sum_res

def train_mode(args):
    os.makedirs( args.saved_model_path, exist_ok=True)
    set_seed(args)
    res1 = test_pair(args)
    res2 = test_single(args)
    json.dump( res1, open(os.path.join(args.saved_model_path, "random_pair.json"), "w") )
    json.dump( res2, open(os.path.join(args.saved_model_path, "test_single.json"), "w") )


if __name__ == "__main__":
     # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')   
    parser.add_argument('--saved_model_path', type = str, default = 'results/mutants_siamese/context', help='filename to output the pre-trained model')  
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--num_class', type = int, default =2, help='num_class')
    parser.add_argument('--seed', type = int, default =0, help='seed')
    parser.add_argument('--task', type=str, default="killed",
                        help='[killed, relevance]')
    parser.add_argument("--projects", nargs="+", default=["collections"])
    parser.add_argument("--test_projects", nargs="+", default=[])
   
  
    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if len(args.test_projects) != 0:
        usedp=[]
        for p in args.projects:
            if p not in args.test_projects:
                usedp.append( p  )
        args.projects = usedp
    assert len(args.projects) == 4
    logger = get_logger(os.path.join(args.saved_model_path, "log.txt"))
    logger.info('start training!')
    train_mode(args)
    logger.info('finishing training!')

