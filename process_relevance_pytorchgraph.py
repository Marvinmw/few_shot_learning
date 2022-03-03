import argparse
import json
from multiprocessing import Pool
from networkx.drawing import nx_agraph
import networkx as nx
from pandas.io import parsers
import pygraphviz
import torch
import numpy as np
from torch_geometric.data import Data
import os.path as osp
import pandas as pd
import collections
import tqdm
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from utils.TokenizerW2V import TokenIns
import os

Declariation=13
VariableDeclaration=3
word2vec_file="tokens/jars/emb_100.txt"
tokenizer_file="tokens/jars/fun.model"

def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # nodes, node feat, node type
    atom_features_list = []
    atom_node_type_list = []
    variable_declar_mask_list = []
    ins_length_list = []
    for _, node in G.nodes(data=True):
        atom_feature = node['feat']
        atom_features_list.append(atom_feature)
        atom_node_type_list.append( node['type'] )
        ins_length_list.append( node["ins_length"] )
        if node['type'] == Declariation:
           variable_declar_mask_list.append( True )
        else:
           variable_declar_mask_list.append( False )  

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long )
    x_type = torch.tensor(np.array(atom_node_type_list), dtype=torch.long )
    ins_length = torch.tensor( np.array( ins_length_list), dtype=torch.long )
    variable_declar_mask = torch.tensor(np.array(variable_declar_mask_list ) , dtype=bool)
    # edges, edge type
    dv_mask_list = []
    if len(G.edges()) > 0:  
        edges_list = []
        edge_label_list = []
        edge_node_type_list = []
        for i, j, edge in G.edges(data=True):
            edge_type =  edge['label'] 
            edges_list.append((i, j))
            edge_node_type_list.append(edge['type'])
            edge_label_list.append(edge_type)
            # edges_list.append((j, i))
            # edge_type_list.append(edge_type)
            if edge_type == VariableDeclaration:
                dv_mask_list.append(True)
            else:
                dv_mask_list.append(False)
                        
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long )
        dv_mask = torch.tensor(np.array(dv_mask_list), dtype=bool)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        # edge_type_list_np = np.array(edge_type_list)
        # print(np.array(edge_type_list))
        edge_attr = torch.tensor(np.array(edge_label_list), dtype=torch.long)
        edge_node_attr =  torch.tensor(np.array(edge_node_type_list), dtype=torch.long)
        #print(edge_attr.size())
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0), dtype=torch.long)
        edge_node_attr = torch.empty((2, 0), dtype=torch.long)
        dv_mask =  torch.empty((0), dtype=torch.long)

    assert edge_index.shape[-1] == len(edge_attr), f"{edge_index}, {edge_attr.shape}, {len(G.nodes()) }"
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_type = x_type
    data.variable_declar_mask = variable_declar_mask
    data.dv_mask = dv_mask
    data.ins_length = ins_length
    return data


     
def preprocess(class_method_id_json, graph_json , rawins_json
    , nodetype, edgetype, tokenizer_word2vec, dataname, graph_ids, datatype, outputdir, mutant_types):
    inputgraph_meta = json.load( open( graph_json ) )
    rawins_meta = json.load( open( rawins_json ) )
    graph_id_list = [ ]
    graph_labels = []
    data_list = []
    print(class_method_id_json)
    for method_id in graph_ids:       
        if method_id not in inputgraph_meta:
            continue        
        graph_string = inputgraph_meta[ method_id ]
        instructions = {}
        if method_id not in rawins_meta:
            print(f"{class_method_id_json}, {method_id} ")
        instructions.update( rawins_meta[ method_id ]["Local"] )
        instructions.update( rawins_meta[ method_id ]["Unit"] )
        g_meta = nx_agraph.from_agraph(pygraphviz.AGraph(graph_string, directed=True))
        if len(g_meta.nodes) == 0:
            continue
        simple_graph = nx.DiGraph()
                
        id_rename_mapping = {}
        for (nid, info) in list( g_meta.nodes(data=True) ):
            ntype = int( nodetype[ info["type"] ] )
            ins = instructions[nid]
            subwordsOfins = tokenizer_word2vec.get_tokens_id(ins.strip())
            feat = subwordsOfins + [ 0 ] * (100 - len(subwordsOfins) ) if len(subwordsOfins) < 100 else subwordsOfins[:100]
            simple_graph.add_node(nid, type= ntype, feat=feat,ins_length = min(100, len(subwordsOfins) ))

        for e1, e2, a in g_meta.edges( data=True ):
            if len( g_meta[e1][e2] ) > 1:
                #print( g_meta[e1][e2] )
                paralle_edges=[ edgetype[ g_meta[e1][e2][i]["label"]] for i in g_meta[e1][e2] ]
                paralle_edges = sorted(paralle_edges)
                if paralle_edges == [0, 1]:
                    etype =  int( edgetype[ "DataDependence|ControlDependence" ] ) 
                if paralle_edges == [0, 2]:
                    etype = int( edgetype[ "Controlfow|DataDependence" ] )
                if paralle_edges == [1, 2]:
                    etype = int( edgetype[ "Controlfow|ControlDependence" ] ) 
                if paralle_edges == [0, 1, 2]:
                    etype= int( edgetype[ "Controlfow|ControlDependence|ControlDependence" ] )
            else:
                    etype = int( edgetype[ a["label"] ] ) 
            simple_graph.add_edge( e1, e2, label=etype, type=[int(simple_graph.nodes[e1]["type"]), int(simple_graph.nodes[e2]["type"])] )
                        

        # rename id from 0 to number of nodes - 1
        counter = 0
        for i in sorted(simple_graph):
            id_rename_mapping[i] = counter
            counter += 1
        simple_graph = nx.relabel_nodes( simple_graph, id_rename_mapping, copy=False )
        data_geometric = nx_to_graph_data_obj_simple( simple_graph )
             
        # print(data_geometric.y )
        data_geometric.graphID = int(method_id) 
        if datatype == "mutants":
            data_geometric.mutantID = int(graph_ids[method_id]["mid"])
            data_geometric.interaction = int(graph_ids[method_id]["interaction"])
            data_geometric.mutant_type = int(mutant_types[graph_ids[method_id]["mutators"]])
            data_geometric.graph_label = int( graph_ids[method_id]["label"] )
            data_geometric.org_graph_id = int( graph_ids[method_id]["org_graph_id"])
            data_geometric.y= int( graph_ids[method_id]["label"] )
            graph_id_list.append(  method_id )
        elif datatype == "original":
            data_geometric.mutantID = -1
            data_geometric.graph_label = -1
            data_geometric.y = -1
            graph_id_list.append(  int(method_id) )
        else:
            assert False
        data_list.append( data_geometric  )
        graph_labels.append( data_geometric.graph_label )
        
        del g_meta
        del simple_graph
    torch.save([data_list,graph_labels, graph_id_list ], osp.join(outputdir, f"{dataname}.pt"))
    return len(data_list)


def preprocess_relevance(pfolder):
    mutant_meta = json.load( open( os.path.join(pfolder, "mutants_info_graph_ids.json") ) )
    mutant_types = json.load( open( os.path.join(pfolder, "mutants_type.json") ) )
    if mutant_meta is None:
        print(f"========================= {pfolder}")
        return
    tokenizer_word2vec = TokenIns(
            word2vec_file=word2vec_file,
            tokenizer_file=tokenizer_file
            )
    nodetype = json.load( open("tokens/instruction_type.json") )
    edgetype = json.load( open("tokens/edge_type.json") )

    mutant2Graph = {}
    org2Mutant = set()
    for mutant_id in mutant_meta:
         mutant_meta[ mutant_id ]["mid"] = mutant_id
         mutant2Graph[mutant_meta[mutant_id]["mutant_graph_id"]] = mutant_meta[ mutant_id ]
         org2Mutant.add( mutant_meta[mutant_id]["org_graph_id"] )

    tasks = { "original":org2Mutant, "mutants":mutant2Graph }
    pid = os.path.basename( pfolder )
    for k,v in tasks.items():
        outputdir = os.path.join( outputfolder, pid, "raw", k, "graph")
        os.makedirs(outputdir, exist_ok=True)
        class_method_id_json = os.path.join( pfolder, k, "class_method_id_mapping.json" )
        rawins_json = os.path.join(pfolder, k, "RawIns.json")
        for dataname in [ "DV_CFG", "DV_PDG" , "ORG_PDG" , "ORG_CFG"]: 
            graph_json = os.path.join(pfolder, k, f"{dataname}.json")
            preprocess(class_method_id_json, graph_json , rawins_json
                , nodetype, edgetype, tokenizer_word2vec, dataname,  v, k,outputdir, mutant_types)

    
def run(data_folder):
    eggs = []
    for project in os.listdir( data_folder ):
        pfolder = os.path.join( data_folder, project )   
        eggs.append( pfolder )
    with Pool(15) as p:
        p.map(preprocess_relevance, eggs)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", default="./dataset/relevant")
    parser.add_argument("-i", "--input", dest="input", default="./dataset/relevant_raw/")
    args = parser.parse_args()
    outputfolder = args.output
    run( args.input  )
    

    
                

                 


