import json
import os
import glob
import shutil
import re
import collections

def remove_prefix(s, prefix):
    return s[len(prefix): ] if s.startswith(prefix) else s

def remove_suffix(s, suffix):
    return s[ : -len(suffix) ] if s.endswith(suffix) else s

import shutil

def extract_mutants_graph_task(input):
    print("****************************************************")
    print(input)
    input = input.strip()
    mutant_info = json.load( open(os.path.join(input, "mutants_info.json") ))
    mutants_graphs = json.load( open(os.path.join(input, "mutants", "class_method_id_mapping.json"), "r") )
    org_graphs = json.load( open(os.path.join( input, "original","class_method_id_mapping.json"), "r") )
       
    index_mutants = {}
    index_org_graph = {} 
    mutant_type = {}
    for k, v in mutants_graphs.items():
        splited_keys = k.strip().split(".")
        id = splited_keys[-2]    # mid
        classname = splited_keys[-1]
        assert id not in index_mutants, f"{id}, {k}"
        index_mutants[ id ] = v
        index_org_graph[ id ] = org_graphs[ classname ]
     

    for mid in mutant_info: # mutant id
        if mutant_info[mid]["mutators"] not in mutant_type:
            mutant_type[ mutant_info[mid]["mutators"].strip() ] = len( mutant_type )
        mutant_position = int(mutant_info[mid]["lineNumbers"]) # line number
        mutant_class_graphs = index_mutants[mid]
        org_class_graphs = index_org_graph[mid]
        found = 0
        #print("Mu "+mutant_info[mid]["mutatedMethod"] + f" {mutant_position}")
        for org_method_graph_id in org_class_graphs: # method id
            start_lno = int(org_class_graphs[org_method_graph_id]["start"]) # bytecode source code line number
            end_lno = int(org_class_graphs[org_method_graph_id]["end"])
            #print("Org "+org_class_graphs[org_method_graph_id]["name"]+f" {start_lno} {end_lno}")
            if mutant_position >= start_lno and mutant_position <= end_lno: # line number location
                mutant_info[ mid ][ "org_graph_id"] =  org_method_graph_id       
                found += 1
                break
        for check_id in mutant_class_graphs:
            s = mutant_class_graphs[check_id]["start"]
            e = mutant_class_graphs[check_id]["end"]
            #print("MG "+mutant_class_graphs[check_id]["name"]+f" {s} {e}")
            if mutant_position >= s and mutant_position <= e: # line number location
                        mutant_info[ mid ][ "mutant_graph_id" ] = check_id
                        found += 1
                        break
        assert found == 2,f"{mid}"
    
    

    json.dump(mutant_info, open(os.path.join(input, "mutants_info_graph_ids.json"), "w" ), indent=6) 
    json.dump(mutant_type, open(os.path.join(input, "mutants_type.json"), "w" ), indent=6)                



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract mutants source code.')
    parser.add_argument( '--input', default="defects4j/res_middle/Chart_12_fixed|defects4j_middle_graph/Chart_12_fixed" , type=str)
    parser.add_argument( '--task', default="extract_mutants_graph_task" , type=str)
    args = parser.parse_args()
    task_fn = {"extract_mutants_graph_task":extract_mutants_graph_task}
    # if args.task in task_fn:
    #     task_fn[args.task]( args.input ) 
    extract_mutants_graph_task("dataset/relevant_raw/fom_dot")
    #extract_mutants_source_code("defects4j/res_middle/JacksonDatabind_15_fixed")
    #mapping_mutants2orginalBinFiles_task("defects4j/res_middle/JacksonDatabind_15_fixed")
    #extract_mutants_graph_task("defects4j/res_middle/JacksonDatabind_15_fixed|defects4j_middle_graph/JacksonDatabind_15_fixed")
