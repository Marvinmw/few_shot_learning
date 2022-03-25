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

def log_bug(mutant, data, msg):
    print(msg)
    json.dump(data, open("debug_info.json", "w"), indent=6)
    json.dump(mutant, open("debug_info_mutant.json", "w"), indent=6)

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

       # assert id not in index_mutants, f"{id}, {k}, {index_mutants[id]}" merge lamda class to the outer class
        if id not in index_mutants:
            index_mutants[ id ] = v
            if classname in org_graphs:
                index_org_graph[ id ] = org_graphs[ classname ]
            else:
                cc = 0
                for i in range(10):
                    name = classname.split("__")[0]+f"__{i}"
                    print(classname)
                    print(name)
                    if name in org_graphs:
                        index_org_graph[ id ] = org_graphs[ name ]
                assert cc==1
        else:
            c=list(index_mutants[id].values())[0]["class"]
            print(f"{id}, {k}, {c}")
            index_mutants[ id ].update(v)
            if classname in org_graphs:
                index_org_graph[ id ].update(org_graphs[ classname ])
            else:
                cc = 0
                for i in range(10):
                    name = classname.split("__")[0]+f"__{i}"
                    print(classname)
                    print(name)
                    if name in org_graphs:
                        index_org_graph[ id ].update(org_graphs[ name ])
                        cc += 1
                assert cc==1
            

    for mid in mutant_info: # mutant id
        #if mutant_info[mid]["mutators"] not in mutant_type:
        #    mutant_type[ mutant_info[mid]["mutators"].strip() ] = len( mutant_type )
        # mutant_position = int(mutant_info[mid]["lineNumbers"]) # line number
        # mutant_method_name = mutant_info[mid]["mutatedMethod"]
        methodDescription = mutant_info[mid]["methodDescription"]
        methodname = mutant_info[mid]["mutatedMethod"]
        matching_name = f"{methodname}{methodDescription}>"
        mutant_class_graphs = index_mutants[mid]
        org_class_graphs = index_org_graph[mid]
        found = 0
        #print("Mu "+mutant_info[mid]["mutatedMethod"] + f" {mutant_position}")
        
        for org_method_graph_id in org_class_graphs: # method id
            # start_lno = int(org_class_graphs[org_method_graph_id]["start"]) # bytecode source code line number
            # end_lno = int(org_class_graphs[org_method_graph_id]["end"])
            #print("Org "+org_class_graphs[org_method_graph_id]["name"]+f" {start_lno} {end_lno}")
            # if mutant_position >= start_lno and mutant_position <= end_lno: # line number location
            #     mutant_info[ mid ][ "org_graph_id"] =  org_method_graph_id       
            #     found += 1
            #     break
            bytecodesignature = org_class_graphs[org_method_graph_id]["bytecodesignature"]
            bytecodesignature= bytecodesignature.split(":")[-1].strip()
            if matching_name == bytecodesignature:
                mutant_info[ mid ][ "org_graph_id"] =  org_method_graph_id 
                found += 1
                
        assert found == 1,log_bug( mutant_info[mid], org_class_graphs, f"{matching_name} error {input} {mid} in original \n")
        for check_id in mutant_class_graphs:
            #s = mutant_class_graphs[check_id]["start"]
            #e = mutant_class_graphs[check_id]["end"]
            #print("MG "+mutant_class_graphs[check_id]["name"]+f" {s} {e}")
            bytecodesignature = mutant_class_graphs[check_id]["bytecodesignature"]
            bytecodesignature=bytecodesignature.split(":")[-1].strip()
            if matching_name == bytecodesignature:
                mutant_info[ mid ][ "mutant_graph_id"] =  check_id 
                found += 1
            # if mutant_position >= s and mutant_position <= e: # line number location
            #             mutant_info[ mid ][ "mutant_graph_id" ] = check_id
            #             found += 1
            #             break
                        
        assert found == 2,log_bug( mutant_info[mid], mutant_class_graphs, f"error {input} {mid} in mutants \n")

    for mid in mutant_info: # mutant id
        intaracted_mutant_id_list = mutant_info[mid]["interaction"] 
        # print(mutant_info[mid])
        # print(intaracted_mutant_id_list)
        if len(intaracted_mutant_id_list) != 0 and mid not in intaracted_mutant_id_list:
            mutant_info[mid]["interaction_mutant_graph_id"] = []
            for intaracted_mutant_id in intaracted_mutant_id_list:
                if intaracted_mutant_id not in mutant_info:
                    continue
                mutant_info[mid]["interaction_mutant_graph_id"].append( mutant_info[intaracted_mutant_id]["mutant_graph_id"] )
        else:
            mutant_info[mid]["interaction_mutant_graph_id"] = []
    
    

    json.dump(mutant_info, open(os.path.join(input, "mutants_info_graph_ids.json"), "w" ), indent=6) 
   # json.dump(mutant_type, open(os.path.join(input, "mutants_type.json"), "w" ), indent=6)                

import pickle
def mutator_multiple_label( data_folder ):
    multiple_label = {}
    mutant_label = {}
    for mf in glob.glob(f"{data_folder}/**/mutants_info.json", recursive=True):
        print(mf)
        mutants_info = json.load(open(mf))
        for mid, info in mutants_info.items():
            mutator = info ["mutators"].strip()
            mutator=mutator.split("_")[0]
            if (mutator, 0) not in multiple_label:
                multiple_label[(mutator, 0)] = len(multiple_label)
            if (mutator, 1) not in multiple_label:
                multiple_label[(mutator, 1)] = len(multiple_label)
            if mutator not in mutant_label:
                mutant_label[mutator] = len(mutant_label)
    print(len(multiple_label))
    c1 = []
    c2 = []
    for mf in glob.glob(f"{data_folder}/**/mutants_info.json", recursive=True):
        print(mf)
        mutants_info = json.load(open(mf))
        for mid, info in mutants_info.items():
            mutator = info ["mutators"].strip()
            mutator=mutator.split("_")[0]
            rlabel = info["relevance_label"]
            killlabel = info["killed_label"]
            info["relevance_mutator_label"]=multiple_label[(mutator, rlabel)] if (mutator, rlabel) in multiple_label else -1
            c1.append( info["relevance_mutator_label"] )
            info["killabel_mutator_label"]=multiple_label[(mutator, killlabel)]
            info["mutator_label"] = mutant_label[mutator]
            c2.append( info["killabel_mutator_label"] )
        json.dump(mutants_info, open(mf, "w"), indent=4)
    pickle.dump(multiple_label, open("multiple_label.pt" , "wb") )
    with open("multiple_label.txt", "w") as f:
        for k, v in multiple_label.items(): 
            f.write(str(v)+" : "+str(k)+"\n")
    s1 = collections.Counter(c1)
    s2 = collections.Counter(c2)
    json.dump(s1, open("relevance_mutator_label_count.json", "w"), indent=4)
    json.dump(s2, open("killabel_mutator_label_count.json", "w"), indent=4)
    json.dump(mutant_label, open("mutator_label.json", "w"), indent=4)
  #  return mutant_label
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract mutants source code.')
   # parser.add_argument( '--input', default="defects4j/res_middle/Chart_12_fixed|defects4j_middle_graph/Chart_12_fixed" , type=str)
    parser.add_argument( '--task', default="extract_mutants_graph_task" , type=str)
    args = parser.parse_args()
    task_fn = {"extract_mutants_graph_task":extract_mutants_graph_task}
    # if args.task in task_fn:
    #     task_fn[args.task]( args.input ) 
    data_folder = "relevance_java_dot_byteinfo/"
    mutator_multiple_label(data_folder)
    #mutator_label = json.load( open("mutator_label.json") )
    for p in os.listdir( data_folder ) :
        for c in os.listdir( os.path.join(data_folder, p) ):
            #try:
                extract_mutants_graph_task(os.path.join(data_folder, p, c))
            #except Exception as e:
             #       print(e)
                #    with open("extract_graph_failed_projects/log.txt", "a") as f:
               #         f.write(str(e))
                    #shutil.move(os.path.join(data_folder, p, c), "extract_graph_failed_projects/")

    #extract_mutants_source_code("defects4j/res_middle/JacksonDatabind_15_fixed")
    #mapping_mutants2orginalBinFiles_task("defects4j/res_middle/JacksonDatabind_15_fixed")
    #extract_mutants_graph_task("defects4j/res_middle/JacksonDatabind_15_fixed|defects4j_middle_graph/JacksonDatabind_15_fixed")
