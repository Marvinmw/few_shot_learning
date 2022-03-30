from .mutantsdataset import MutantKilledDataset, MutantRelevanceDataset, MutantTestRelevanceDataset
from tqdm import tqdm

def fetch_datalist(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance" or args.task == "subsuming":
            dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p, probability=0.0 )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list

def fetch_testdata(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance" or args.task == "subsuming":
            dataset_inmemory = MutantTestRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list

import json
import collections
import os
import glob
def projects_dict(args):
    projects = collections.defaultdict(list)
    name=[]
    empty_data = json.load( open("dataset/empty_data.json") ) if args.task == "relevance" else []
    if len(args.projects) > 1:
        for p in args.projects:
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
                if os.path.basename(pf) in empty_data:
                    continue
                projects[p].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    elif len(args.projects) == 1:
        for p in args.projects:
            print(p)
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
               # print(pf)
                n=os.path.basename(pf)
                #print(n)
                if os.path.basename(pf) in empty_data:
                    continue
                projects[n].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    return projects, name