import collections
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
import tqdm 
import numpy as np
from itertools import compress
from utils.tools import  inverse_eage
from collections import Counter
import json
import pickle

class MutantsReDataset(InMemoryDataset):
    def __init__(self, root, dataname, project="", transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.project = project
        super(MutantsReDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.pair_data = torch.load(self.processed_paths[0])
        [ self.pairgraph_labels, self.mutants_splitting ] =  torch.load(self.processed_paths[1])

    def split( self, reshuffle=False, binary=False, fixed_size=50, splitting_ratio=-1 ):
        if reshuffle or ( not os.path.isfile( os.path.join(self.root,  "data", self.project, f"{fixed_size}_train.csv.npy") )):
            if not os.path.isdir( os.path.join(self.root,  "data", self.project) ):
                os.makedirs(  os.path.join(self.root,  "data", self.project) )
            data = []
            label=[]
            mid = []
            for l in self.pair_data:
                data.extend(self.pair_data[l])
                mid.extend(self.mutants_splitting[l])
                if int(l) == 0:
                    label.extend([0]*len(self.pair_data[l]))
                    print(f"Label 0 {len(self.pair_data[l])}")
                else:
                    label.extend([1]*len(self.pair_data[l]))
                    print(f"Label 1 {len(self.pair_data[l])}")
                    
            self.data_size = len(data)
            self.data = data
            self.y = label
            statistic = Counter(label)
            print(statistic)
            self.mids = mid
          
            if splitting_ratio > 0:
                indexes = np.arange( len(self.data) )
                np.random.shuffle(indexes)
                test_size = int(splitting_ratio*self.data_size)
                val_size =  int(0.1*self.data_size)
                test_idx = indexes[:test_size]
                valid_idx = indexes[test_size: test_size+val_size]
                train_idx = indexes[test_size+val_size:]
            else:
                indexes = np.arange( len(self.data) )
                np.random.shuffle(indexes)       
                val_size =  int( (self.data_size - fixed_size)*0.2 )
                test_size = self.data_size - val_size - fixed_size

            test_idx = indexes[:test_size]
            valid_idx = indexes[test_size: test_size+val_size]
            train_idx = indexes[:fixed_size]

           
            pickle.dump(test_idx, open( os.path.join(self.root, "data", self.project, f"{fixed_size}_test.csv.npy") , "wb"))
            pickle.dump(valid_idx, open( os.path.join(self.root, "data",self.project, f"{fixed_size}_valid.csv.npy") , "wb"))
            pickle.dump(train_idx, open( os.path.join(self.root, "data", self.project, f"{fixed_size}_train.csv.npy") , "wb"))
            pickle.dump(self.mids, open( os.path.join(self.root,"data", self.project, f"{fixed_size}_mids.csv.npy") , "wb"))
            pickle.dump([self.data, self.y, self.mids], open( os.path.join(self.root,"data", self.project, f"{fixed_size}_sampled_mids.csv.npy") , "wb"))
            json.dump(statistic, open( os.path.join(self.root,"data",self.project,  f"{fixed_size}_statistic.json") , "w"))
        else:
            #print("Load from the previsou splitting")
            [self.data, self.y, self.mids] = pickle.load( open( os.path.join(self.root,"data",self.project,  f"{fixed_size}_sampled_mids.csv.npy") , "rb") )
            

            train_idx =  pickle.load( open( os.path.join(self.root,"data",self.project,  f"{fixed_size}_train.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'train.csv'),  header = None).values.T[0]
            valid_idx = pickle.load( open( os.path.join(self.root,"data",self.project,  f"{fixed_size}_valid.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'valid.csv'),  header = None).values.T[0]
            test_idx = pickle.load( open( os.path.join(self.root,"data",self.project,  f"{fixed_size}_test.csv.npy") , "rb") )#pd.read_csv(osp.join(path, 'test.csv'),  header = None).values.T[0]
        
        labels = [ d.y.item() for d in self.data ]
        statistic = Counter(labels)
       # print(statistic)        
        #print(f"Train Data Size {len(train_idx)}, Valid Data Size {len(valid_idx)}, Test Data Size {len(test_idx)}, Num {len(self.mids)}")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
           
    
    def get(self, idx):
        data = Data()
        pass
        
    @property
    def raw_dir(self):
        return os.path.join(self.root)
        
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""  
        return self.raw_file_names[0] + self.raw_file_names[1]

    @property
    def raw_file_names(self):
        mutant_file_name_list = []
        original_file_name_list = []
        mutant_file_name_list.append(  os.path.join( self.root ,f"{self.project}", "raw", "mutants", "graph", f"{self.dataname}.pt")   )
        original_file_name_list.append(  os.path.join( self.root ,f"{self.project}", "raw", "original", "graph", f"{self.dataname}.pt")   )
        return mutant_file_name_list, original_file_name_list
    
    @property
    def processed_file_names(self):
        return [ f'geometric_data_processed_{self.dataname}_{self.project}.pt', f"data_info_{self.project}.pt"]
    
    def download(self):
        pass
    
    def orginal_grap_mapping(self, ofile):
        org_data = torch.load(os.path.join( ofile ))
        org_data_list, _, org_data_id = org_data[0], org_data[1], org_data[2]
        org_data_list = [ inverse_eage(d) for d in org_data_list ]
        org = { d.graphID:d for d in org_data_list }
        return org


    def process(self):        
       # mutanttyper=json.load( open( os.path.join(os.path.join(self.root, "mutant_type.json")) ) )    
        pairgraph_labels = []
        pair_data_list = collections.defaultdict(list)
        mutant_file_name_list, original_file_name_list = self.raw_file_names

        mutants_splitting = collections.defaultdict(list)
        for file_id in tqdm.tqdm( range(len(mutant_file_name_list)) ):
            mfile = mutant_file_name_list[file_id]
            pname = mfile.split("/")[-4]
            ofile = original_file_name_list[file_id]
            mutant_data = torch.load(os.path.join( mfile ))
            mutant_data_list, graph_labels, graph_ids = mutant_data[0], mutant_data[1], mutant_data[2]
            mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
            org_data = self.orginal_grap_mapping( ofile )
            for mutant_graph in mutant_data_list:
                    mid=mutant_graph.mutantID
                    ogid=mutant_graph.org_graph_id
                    orggraph=org_data[ogid]
                    pair_data_list[mutant_graph.graph_label].append(PairData(mutant_graph.edge_index,  mutant_graph.x, mutant_graph.edge_attr,mutant_graph.ins_length, 
                                                    orggraph.edge_index,  orggraph.x, orggraph.edge_attr, orggraph.ins_length, 
                                                    torch.tensor(mutant_graph.graph_label), torch.tensor(mutant_graph.mutant_type) ))
                 
                    mutants_splitting[mutant_graph.graph_label].append( f"{pname}_{mid}"  )
                       
        self.data_size = len(pair_data_list)                        
        torch.save(pair_data_list, self.processed_paths[0])
        torch.save( [ pairgraph_labels, mutants_splitting ], self.processed_paths[1] )

def balanced_subsample(x,y,mid_list,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        mid =  list(compress(mid_list, idx )) 
        class_xs.append((yi, elems, mid))
        print(f"label {yi}, Number {len(elems)}")
        if min_elems == None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []
    mids= []
    for ci,this_xs, this_mids in class_xs:
        index = [i for i in range(len(this_xs))]
        if len(this_xs) > use_elems:
            np.random.shuffle(index)

        x_ = [ this_xs[i] for i in index[:use_elems] ]  #this_xs[:use_elems]
        mid_ = [ this_mids[i] for i in index[:use_elems] ]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.extend(x_)
        ys.extend(y_.tolist())
        mids.extend(mid_)

    return xs       

import random
def balanced_oversample(x, y):
    class_xs = []
    max_elems=None
    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        class_xs.append((yi, elems))
        print(f"label {yi}, Number {len(elems)}")
        if max_elems == None or len(elems) > max_elems:
            max_elems = len(elems)

    use_elems = max_elems
  
    xs = []
    ys = []

    for ci,this_xs in class_xs:
        index = [i for i in range(len(this_xs))]
        if use_elems > len(this_xs):
            index = random.choices(index, k=use_elems)

        x_ = [ this_xs[i] for i in index ]  #this_xs[:use_elems]
        y_ = np.empty(use_elems,  dtype=int)
        y_.fill(ci,)
        
        xs.extend(x_)
        ys.extend(y_.tolist())

    for yi in np.unique(ys):
        idx =  [ id ==yi for id in ys ]
        elems = list(compress(xs, idx )) 
        print(f"label {yi}, Number {len(elems)}")
     
    return xs      

class PairData(Data):
    
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s=None, ins_length_s=None, edge_index_t=None, x_t=None,  edge_attr_t=None, ins_length_t=None,y=None, type=None, operand1=None, operand2=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_attr_s = edge_attr_s
        self.ins_length_s = ins_length_s
            
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.ins_length_t = ins_length_t
        self.y = y
        self.type=type
        self.operand1=operand1
        self.operand2=operand2
        if x_s is None:
             print("Debug")

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


      
    





    

 
  
