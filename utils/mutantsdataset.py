from cProfile import label
import json
import collections
from mimetypes import init
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
import numpy as np
from itertools import compress
from .tools import  inverse_eage
import numpy as np
from .tools import get_logger

import torch
from torch_geometric.data import Dataset

class MutantKilledDataset(InMemoryDataset):
    def __init__(self, root, dataname, project="", transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.project = project
        super(MutantKilledDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data = torch.load(self.processed_paths[0])
        self.data_folder = os.path.dirname( self.processed_paths[0] )
        [ self.kill_mutant_binary_labels, self.kill_mutant_multiple_labels ] =  torch.load(self.processed_paths[1])

  
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
        mutant_file_name =  os.path.join( self.root , "raw", "mutants", "graph", f"{self.dataname}.pt")   
        original_file_name =  os.path.join( self.root , "raw", "original", "graph", f"{self.dataname}.pt")   
        return mutant_file_name, original_file_name
    
    @property
    def processed_file_names(self):
        return [ f'killed/kill_processed_{self.dataname}_{self.project}.pt', f"killed/kill_info_{self.project}.pt",
        f"killed/bstat_{self.dataname}_{self.project}.json", f"killed/mstat_{self.dataname}_{self.project}.json"]
    
    def download(self):
        pass
    
    def splitting_ratio(self):
        data_size = len(self.data)
        index = [ i for i in range(data_size) ]
        random.shuffle(index)
        test_size = int( 0.2 * data_size )
        val_size = int( 0.2 * (data_size - test_size) )
        test_index = index[ : test_size ]
        val_index = index[ test_size : test_size+val_size]
        train_index = index[ test_size+val_size :  ]
        json.dump( {"test":test_index, "val": val_index, "train": train_index}, open( os.path.join(self.data_folder , "splitting_ratio.json") , "w") , indent=6)
        return len(test_index),len(val_index),len(train_index)
    
    def splitted_fixed(self, label_list, fixed_train=10):
        data_size = len(self.data)
        index = [ i for i in range(data_size) ]
        random.shuffle(index)
        unique_labels = list(set( label_list ))
        num_class = len(unique_labels)
        split_dict = {}
        for label in unique_labels:
            bool_index = [ i==label for i in label_list ]
            selected_data = list(compress( index, bool_index ))
            split_dict[label] = selected_data
        
        train_index = []
        
        for l in split_dict:
            d = split_dict[l]
            train_index.extend( d[:fixed_train] )
        
        remaining_index = [ ]
        for i in index:
            if i not in train_index:
                remaining_index.append(i)

        val_size = int( len( remaining_index ) * 0.4 )
        val_index = remaining_index[ : val_size ]
        test_index = remaining_index[val_size:]
        json.dump( {"test":test_index, "val": val_index, "train": train_index}, open( os.path.join(self.data_folder ,f"splitting_fixed_{num_class}.json") , "w") , indent=6 )
        return len(test_index),len(val_index),len(train_index)

    def process(self):        
        # mutanttyper=json.load( open( os.path.join(os.path.join(self.root, "mutant_type.json")) ) )    
        kill_mutant_binary_labels = []
        kill_mutant_multiple_labels = []
        kill_mutant_data = []
        mutant_file_name, original_file_name= self.raw_file_names
        mfile = mutant_file_name
        ofile = original_file_name
        mutant_data = torch.load(os.path.join( mfile ))
        mutant_data_list, _ = mutant_data[0], mutant_data[1]
        mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
        org_data = torch.load(os.path.join( ofile ))
        org_data_list, _ = org_data[0], org_data[1]
        org_data_list = [ inverse_eage(d) for d in org_data_list ]
        original_graph_dict = {  }
        
        for g in org_data_list:
            original_graph_dict[ g.graphID  ] = g
        
        for mutant_graph in mutant_data_list:
                mid = mutant_graph.mutantID
                kill_label = mutant_graph.label_k_binary
                origina_graph_id = mutant_graph.org_graph_id
                org_graph = original_graph_dict[origina_graph_id]
                if kill_label == -1: # not considered
                    continue
                kill_mutant_data.append(PairData(mutant_graph.edge_index,  mutant_graph.x, mutant_graph.edge_attr,mutant_graph.ins_length, 
                                                            org_graph.edge_index,  org_graph.x, org_graph.edge_attr, org_graph.ins_length, 
                                                            torch.tensor(mutant_graph.label_k_binary), torch.tensor(mutant_graph.label_k_mul), torch.tensor(mutant_graph.submsuing), torch.tensor(mutant_graph.mutant_type), torch.tensor(mid) ))
                kill_mutant_binary_labels.append( mutant_graph.label_k_binary )
                kill_mutant_multiple_labels.append( mutant_graph.label_k_mul )
        print(  os.path.dirname(self.processed_paths[0]) )                                               
        if not os.path.isdir( os.path.dirname(self.processed_paths[0]) ):
           os.makedirs( os.path.dirname(self.processed_paths[0]), exist_ok=True )                 
        torch.save(kill_mutant_data, self.processed_paths[0])
        torch.save( [ kill_mutant_binary_labels, kill_mutant_multiple_labels ], self.processed_paths[1] )
        bstat = collections.Counter(kill_mutant_binary_labels)
        mstat = collections.Counter(kill_mutant_multiple_labels)
        json.dump(bstat, open(self.processed_paths[2], "w") , indent=6)
        json.dump(mstat, open(self.processed_paths[3], "w") , indent=6)


#logger = get_logger("empty_data.txt", init=True)

class MutantRelevanceDataset(InMemoryDataset):
    def __init__(self, root, dataname, project="",probability=0.6, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.project = project
        self.probability = probability
        super(MutantRelevanceDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        self.data = torch.load(self.processed_paths[0])
        self.data_folder = os.path.dirname( self.processed_paths[0] )
        [ self.relevance_mutant_binary_labels, self.relevance_mutant_multiple_labels ] =  torch.load(self.processed_paths[1])
        

  
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
        mutant_file_name =  os.path.join( self.root , "raw", "mutants", "graph", f"{self.dataname}.pt")   
        original_file_name =  os.path.join( self.root , "raw", "original", "graph", f"{self.dataname}.pt")   
        return mutant_file_name, original_file_name
    
    @property
    def processed_file_names(self):
        suffix=f"inmethod_{self.dataname}_{self.project}_{self.probability}"
        return [ f'relevance/relevance_processed_{suffix}.pt', f"relevance/relevance_info_{suffix}.pt", 
                f"relevance/bstat_{suffix}.json", f"relevance/mstat_{suffix}.json"]
    
    def download(self):
        pass
    
    def splitting_ratio(self):
        data_size = len(self.data)
        index = [ i for i in range(data_size) ]
        random.shuffle(index)
        test_size = int( 0.2 * data_size )
        val_size = int( 0.2 * (data_size - test_size) )
        test_index = index[ : test_size ]
        val_index = index[ test_size : test_size+val_size]
        train_index = index[ test_size+val_size :  ]
        json.dump( {"test":test_index, "val": val_index, "train": train_index}, open( os.path.join(self.data_folder , f"{self.probability}_splitting_ratio.json") , "w") , indent=6)
        return len(test_index),len(val_index),len(train_index)
    
    def splitted_fixed(self, label_list, fixed_train=10):
        data_size = len(self.data)
        index = [ i for i in range(data_size) ]
        random.shuffle(index)
        unique_labels = list(set( label_list ))
        num_class = len(unique_labels)
        split_dict = {}
        for label in unique_labels:
            bool_index = [ i==label for i in label_list ]
            selected_data = list(compress( index, bool_index ))
            split_dict[label] = selected_data
        
        train_index = []
        
        for l in split_dict:
            d = split_dict[l]
            train_index.extend( d[:fixed_train] )
        
        remaining_index = [ ]
        for i in index:
            if i not in train_index:
                remaining_index.append(i)

        val_size = int( len( remaining_index ) * 0.4 )
        val_index = remaining_index[ : val_size ]
        test_index = remaining_index[val_size:]
        json.dump( {"test":test_index, "val": val_index, "train": train_index}, open( os.path.join(self.data_folder ,f"{self.probability}_splitting_fixed_{num_class}.json") , "w") , indent=6 )
        return len(test_index),len(val_index),len(train_index)

    def process(self):        
        # mutanttyper=json.load( open( os.path.join(os.path.join(self.root, "mutant_type.json")) ) )    
        mutant_meta_file =  os.path.join( self.root , "raw", "mutants", "mutants_info_graph_ids.json")   
        mutant_meta = json.load(open(mutant_meta_file))
        relevance_mutant_binary_labels = []
        relevance_mutant_multiple_labels = []
        relevance_mutant_data = []
        mutant_file_name, _ = self.raw_file_names
        mfile = mutant_file_name
        mutant_data = torch.load(os.path.join( mfile ))
        mutant_data_list, _ = mutant_data[0], mutant_data[1]
        mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
       
        graph_dict = {  }
        on_change_graph = { } 
        change_mid_list = []
        for g in mutant_data_list:
            if g.on_change:
                on_change_graph[ g.mutantID ] = g
                change_mid_list.append( g.mutantID )
            else:
                graph_dict[ g.mutantID  ] = g
        

        for mid in graph_dict:
                mutant_graph = graph_dict[mid]
                relevance_label = mutant_graph.label_r_binary
                if relevance_label == -1: # not considered
                    continue
                interacted_mid_list = mutant_meta[str(mid)]["interaction"]
                mutant_method_name =  mutant_meta[str(mid)]["mutatedMethod"]
                mutant_class_name = mutant_meta[str(mid)]["mutatedClass"]
                interacted_mid_list = [ int(i) for i in interacted_mid_list]
                pos_graph = None
                
                if len(interacted_mid_list) != 0:
                    for interacted_mid in interacted_mid_list:
                        # print(mid)
                        # print(interacted_mid_list)
                        # print(on_change_graph.keys())
                        if mutant_meta[str(interacted_mid)]["mutatedMethod"] == mutant_method_name and mutant_meta[str(interacted_mid)]["mutatedClass"] == mutant_class_name:
                            pos_graph = on_change_graph[ int(interacted_mid) ] 
                            relevance_mutant_data.append(PairData(mutant_graph.edge_index,  mutant_graph.x, mutant_graph.edge_attr,mutant_graph.ins_length, 
                                                                    pos_graph.edge_index,  pos_graph.x, pos_graph.edge_attr, pos_graph.ins_length, 
                                                                    torch.tensor(mutant_graph.label_r_binary), torch.tensor(mutant_graph.label_r_mul), torch.tensor(mutant_graph.submsuing_r), torch.tensor(mutant_graph.mutant_type), torch.tensor(mid) ))
                            relevance_mutant_binary_labels.append( mutant_graph.label_r_binary )
                            relevance_mutant_multiple_labels.append( mutant_graph.label_r_mul )
                remaining_graph_list = []
                for cid in on_change_graph:
                    if cid not in  interacted_mid_list:
                        if mutant_meta[str(cid)]["mutatedMethod"] == mutant_method_name and mutant_meta[str(cid)]["mutatedClass"] == mutant_class_name:
                            remaining_graph_list.append( on_change_graph[cid] ) 

                #candiates_list = neg_graph_list
                random.shuffle(remaining_graph_list)
                first=True
                for c_graph in remaining_graph_list:
                    r = np.random.uniform(0, 1, size=1)
                    if first or r[0] > self.probability:
                        first = False
                        
                        relevance_mutant_binary_labels.append (0 )
                        ml = None
                        if mutant_graph.label_r_mul % 2 == 1:
                            ml = mutant_graph.label_r_mul - 1
                            relevance_mutant_multiple_labels.append( mutant_graph.label_r_mul - 1 )
                        else:
                            ml = mutant_graph.label_r_mul 
                        relevance_mutant_multiple_labels.append( ml  )

                        relevance_mutant_data.append(PairData(mutant_graph.edge_index,  mutant_graph.x, mutant_graph.edge_attr,mutant_graph.ins_length, 
                                                                c_graph.edge_index,  c_graph.x, c_graph.edge_attr, c_graph.ins_length, 
                                                                torch.tensor( 0 ), torch.tensor(ml),torch.tensor(0), torch.tensor(mutant_graph.mutant_type), torch.tensor(mid) ))
        print(  os.path.dirname(self.processed_paths[0]) )
        if not os.path.isdir( os.path.dirname(self.processed_paths[0]) ):
           os.makedirs( os.path.dirname(self.processed_paths[0]), exist_ok=True )          
        torch.save(relevance_mutant_data, self.processed_paths[0])
        torch.save( [ relevance_mutant_binary_labels, relevance_mutant_multiple_labels ], self.processed_paths[1] )
        bstat = collections.Counter(relevance_mutant_binary_labels)
        mstat = collections.Counter(relevance_mutant_multiple_labels)
        json.dump(bstat, open(self.processed_paths[2], "w") , indent=6)
        json.dump(mstat, open(self.processed_paths[3], "w") , indent=6)
        # if len(relevance_mutant_binary_labels) == 0:
        #     logger.info(f"{self.root}")


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        #mid =  list(compress(mid_list, idx )) 
        class_xs.append((yi, elems))
        print(f"label {yi}, Number {len(elems)}")
        if min_elems == None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []
    #mids= []
    for ci,this_xs in class_xs:
        index = [i for i in range(len(this_xs))]
        if len(this_xs) > use_elems:
            np.random.shuffle(index)

        x_ = [ this_xs[i] for i in index[:use_elems] ]  #this_xs[:use_elems]
        #mid_ = [ this_mids[i] for i in index[:use_elems] ]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.extend(x_)
        ys.extend(y_.tolist())
        #mids.extend(mid_)

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

    use_elems = min(2000, max_elems)
  
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



class MutantTestRelevanceDataset(Dataset):
    def __init__(self, root, dataname, project="", transform=None, pre_transform=None, pre_filter=None):
            self.root = root
            self.dataname = dataname
            self.project = project
            super(MutantTestRelevanceDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
            [ self.considered_mutants, self.change_mutant_index, self.interaction_ground ] = torch.load(self.processed_paths[0])
            mutant_file_name, _ = self.raw_file_names
            mutant_data = torch.load(os.path.join( mutant_file_name ))
            mutant_data_list, _ = mutant_data[0], mutant_data[1]
            self.mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
            self.data_folder = os.path.dirname( self.processed_paths[0] )
            self.set_bank()
            self.set_query_mutants()
    
    def set_bank(self):
        self.bank = [ self.mutant_data_list[i] for i in self.change_mutant_index ]
    
    def prepare_fine_tune(self):

        pass

    def set_query_mutants(self):
        self.query_mutants = [ self.mutant_data_list[i] for i in self.considered_mutants ] #list( compress(self.mutant_data_list, self.considered_mutants ) )
        
    def set_data(self,key):
        if key == "bank":
            self.data = self.bank
        elif key == "query":
            self.data = self.query_mutants
        else:
            assert False, f"key error {key}"

    def get(self, idx):
        data = self.data[idx]
        return data

    def len(self):
        return len(self.data)

    @property
    def raw_dir(self):
        return os.path.join(self.root)
        
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""  
        return self.raw_file_names[0] + self.raw_file_names[1]

    @property
    def raw_file_names(self):
        mutant_file_name =  os.path.join( self.root , "raw", "mutants", "graph", f"{self.dataname}.pt")   
        original_file_name =  os.path.join( self.root , "raw", "original", "graph", f"{self.dataname}.pt")   
        return mutant_file_name, original_file_name
    
    @property
    def processed_file_names(self):
        suffix=f"inmethod_{self.dataname}_{self.project}_idpair"
        return [ f'relevance/relevance_processed_{suffix}.pt']
    
    def download(self):
        pass
    

    def process(self):        
        # mutanttyper=json.load( open( os.path.join(os.path.join(self.root, "mutant_type.json")) ) )    
        mutant_meta_file =  os.path.join( self.root , "raw", "mutants", "mutants_info_graph_ids.json")   
        mutant_meta = json.load(open(mutant_meta_file))
        mutant_file_name, _ = self.raw_file_names
        mfile = mutant_file_name
        mutant_data = torch.load(os.path.join( mfile ))
        mutant_data_list, _ = mutant_data[0], mutant_data[1]
        mutant_data_list = [ inverse_eage(d) for d in mutant_data_list ]
        
        graph_dict = {  }
        on_change_graph = { } 
        change_mutant_index = []
        for k, g in enumerate(mutant_data_list):
            if g.on_change:
                on_change_graph[ g.mutantID ] = ( g, k )
                change_mutant_index.append(  k  )
            else:
                graph_dict[ g.mutantID  ] =  ( g, k )
        
        considered_mutants_index = []
        interaction_index = {}
        for mid in graph_dict:
                (mutant_graph, mutant_graph_index ) = graph_dict[mid]
                relevance_label = mutant_graph.label_r_binary
                if relevance_label == -1: # not considered
                   # print("not consider")
                    continue
                
                interacted_mid_list = mutant_meta[str(mid)]["interaction"]
                mutant_method_name =  mutant_meta[str(mid)]["mutatedMethod"]
                mutant_class_name = mutant_meta[str(mid)]["mutatedClass"]
                interacted_mid_index_list = []
                for interacted_mid in interacted_mid_list:
                    if mutant_meta[str(interacted_mid)]["mutatedMethod"] == mutant_method_name and mutant_meta[str(interacted_mid)]["mutatedClass"] == mutant_class_name:
                        interacted_mid_index_list.append( on_change_graph[int(interacted_mid)][1] )
                        considered_mutants_index.append( mutant_graph_index )
                        interaction_index[ mutant_graph_index ] = interacted_mid_index_list
             
                
        #print(  os.path.dirname(self.processed_paths[0]) )
        if not os.path.isdir( os.path.dirname(self.processed_paths[0]) ):
           os.makedirs( os.path.dirname(self.processed_paths[0]), exist_ok=True )          
        torch.save( [ considered_mutants_index, change_mutant_index, interaction_index ], self.processed_paths[0])
        #pos_index = [ ]
        #neg_index = [ ]
        #for k in interaction_index:
        #    if len(interaction_index[k]) > 0:
        #        pos_index.append( k )
        #    else:
        #        neg_index.append( k )
        #random.shuffle( k_list )

          

class PairData(Data):
    
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s=None, ins_length_s=None, edge_index_t=None, x_t=None,  edge_attr_t=None, 
                        ins_length_t=None,by=None,my=None, sy=None,type=None, mid=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_attr_s = edge_attr_s
        self.ins_length_s = ins_length_s
            
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.ins_length_t = ins_length_t
        self.by = by
        self.my = my
        self.type=type
        self.mid = mid
        self.sy = sy
        if x_s is None:
             print("Debug")

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


      
