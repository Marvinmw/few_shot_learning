from argparse import ArgumentParser
import json
import collections
import pandas as pd
import glob
import os
from pathlib import Path
import shutil
import pickle 

def rename_mutantfoldername(mutatns_folder):
     # rename folder
    flist = [f for f in  glob.glob(f"{mutatns_folder}/**/details.txt" , recursive=True)]
    counter = 0
    for f in flist:
        mp = Path(os.path.dirname(f))
        shutil.move(str(mp),  str(Path(mp.parent, f"{counter}_tmp")))   
        print( str(mp))
        print(  str(Path(mp.parent, f"{counter}_tmp")) )
        counter += 1
    for f in glob.glob(f"{mutatns_folder}/**/details.txt" , recursive=True): 
        mp = str(Path(os.path.dirname(f)))
        nmp = mp.replace("_tmp", "")
        shutil.move(mp,  nmp)   

def copy_original_class(commit_folder):
    os.makedirs(f"{commit_folder}/original/", exist_ok=True )
    for d in glob.glob(f"{commit_folder}/fom_export/**/mutants" , recursive=True):
        if os.path.isdir(d):
            class_name = os.path.basename( Path(d).parent )+".class"
            cpath = str(Path(d).parent.parent).replace("fom_export", "compiled_source/classes")
            orgclasspath = os.path.join(cpath, class_name)
            if not os.path.isfile(orgclasspath):
                print(orgclasspath)
            shutil.copy(orgclasspath, f"{commit_folder}/original/{class_name}" )




def replace_name(mutatns_folder, interactionfile,mutants_info, mutant_matrix_file,outputfolder):
    df = pd.read_csv(mutants_info, index_col=["sourceFile", "lineNumber","index", "block", "mutator"])
    df_kill_matrix = pd.read_csv(mutant_matrix_file, index_col="MutantID")
    df_kill_matrix["sum"] = df_kill_matrix.sum(axis=1)
   # result = df.index.is_unique
   # reid= df[df.index.duplicated()]
    # reid.to_csv("check_id.csv")
    # print(mutants_info)
  #  assert result, reid.to_csv("check_id.csv")
    interaction = pd.read_csv(interactionfile)
    interactiondata = interaction.to_dict('records')
    interaction_mapping = {}
    for r in interactiondata:
        interaction_mapping[r['mutant_outside_line_id']] = r['mutant_on_line_id']
    # df.rename({"MutantID":"ind"}, inplace=True)
    # df.rename({"lineNumber":"lineNumbers"}, inplace=True)
    # df.rename({"index":"indexes"}, inplace=True)
    # df.rename({"block":"blocks"}, inplace=True)
    # df.rename({"mutator":"mutators"}, inplace=True)
    mutant_info = collections.defaultdict()
    sum_info = {}
    mid_rowid_mapping = {}
    mid_rowid_mapping_reverse = {}
    for f in glob.glob(f"{mutatns_folder}/**/details.txt" , recursive=True):    
        mid = os.path.basename( os.path.dirname(f) )
        class_name = os.path.basename( Path(f).parent.parent.parent ).split("$")[0]+".java"

        with open(f, "r") as ff:
            info_raw = ff.read().replace('\n', ' ')
            info_raw=info_raw.replace("MutationDetails", "")
            info_raw=info_raw.strip()
            info_raw = info_raw[1:-1]
            info_raw = info_raw.replace("=", ":")
            info_raw = info_raw.replace("[]", "''")
            info_raw = info_raw.replace("[", "")
            info_raw = info_raw.replace("]", "")
            info_raw = info_raw.replace("\s", "")
            items=info_raw.split(",")
            info = {}
            for i in items:
                it = i.split(":")
                info[it[0].strip()]=it[-1].strip()
            info["sourceFile"] = class_name.strip()
            sum_info[f] = info
            
            #print(f)
            #print(df.index)
            if ( info["sourceFile"],int(info["lineNumbers"]), int(info["indexes"]), int(info["blocks"]), info["mutators"] ) in df.index:
                r = df.loc[(info["sourceFile"], int(info["lineNumbers"]), int(info["indexes"]), int(info["blocks"]), info["mutators"] ), :]
              #  r.to_csv("checki_id.csv")
             #   print(info)
                if r["Relevant"] + r["Not_relevant"] + r["On_Change"]:
                    info["relevance_label"] = 1 if r["On_Change"] + r["Relevant"] > 0 else 0
                else:
                    info["relevance_label"] = -1
                rid = r["MutantID"]
                info["killed_label"] = 1 if df_kill_matrix.loc[rid, :]["sum"] > 0 else 0
                info["methodDescription"] = r["methodDescription"]
                info["mutatedMethod"] = r["mutatedMethod"]
                info["mutatedClass"] = r["mutatedClass"]
                info["On_Change"] = str(r["On_Change"])
                info["row_id"] = str(rid)
                # info["sourceFile"] = r["sourceFile"]
                mid_rowid_mapping[mid] = rid
                mid_rowid_mapping_reverse[rid] = mid
                mutant_info[mid] = info
            # if "org.pitest.mutationtest.engine.gregor.mutators.ReturnValsMutator" == info["mutators"] and int(info["lineNumbers"]) == 288:
            #     print(f)
            #     print(info)
            
    
    json.dump(sum_info, open(f"{outputfolder}/all_mutants_details.json", "w"), indent=6 )
    for mid in mutant_info:
        if mutant_info[mid]["relevance_label"] == 1:
            if int(mutant_info[mid]["On_Change"]):
                mutant_info[mid]["interaction"] = mid
                continue
            rid = mid_rowid_mapping[mid] 
            interact_rid = interaction_mapping[rid]
            interact_mid = mid_rowid_mapping_reverse[interact_rid]
            mutant_info[mid]["interaction"] = interact_mid
        else:
            mutant_info[mid]["interaction"] = -1

    for f in glob.glob(f"{mutatns_folder}/**/**.class" , recursive=True):        
        p = Path(f)
        fname = os.path.basename( p )
        name = ".".join( fname.split(".")[-2:] )
        p.rename(Path(p.parent, f"{name}"))            
                
    json.dump(mutant_info, open(f"{outputfolder}/mutants_info.json", "w"), indent=6 )
            
import argparse        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="multiple_label",help="rename [rename mutants folder and commit folder], format [ format .class file name]")
    args = parser.parse_args()
    #rename folder, for each commits, the mutant folder start with from 0 to N
    if args.task == "rename":
        data_folder = "relevance_java_bytecode"
        for p in os.listdir(data_folder):
            for c in os.listdir( os.path.join(data_folder, p) ):
                if not c.endswith(".csv"):
                    c_folder = os.path.join(data_folder, p, c, "fom_export")
                    rename_mutantfoldername(c_folder)

        #rename commits, replace commit hash with number id
        data_folder = "relevance_java_bytecode"
        for p in os.listdir(data_folder):
            pname = p.split("-")[-1]
            counter = 1
            mapping = {}
            for c in os.listdir( os.path.join(data_folder, p) ):
                if c.endswith(".csv"):
                    continue
                mapping[c] = f"{pname}_{counter}"
                shutil.move(str(os.path.join(data_folder, p, c)),  str(os.path.join(data_folder, p, f"{pname}_{counter}")))   
                counter = counter + 1
            json.dump(mapping, open(os.path.join(data_folder, p, "name_mapping.json"), "w"), indent=6 )

    if args.task == "format":
        #rename .class file name, rename class file to standard file name
        data_folder = "relevance_java_bytecode"
      #  data_folder = "debug"
        for p in os.listdir(data_folder):
            for c in os.listdir( os.path.join(data_folder, p) ):
                c_folder = os.path.join(data_folder, p, c)
            #  print(c_folder)
                if os.path.isfile(c_folder):
                    continue
                try:
                    print(f"{c_folder}")
                    replace_name(f"{c_folder}", f"{c_folder}/interaction_pairs.csv",f"{c_folder}/mutants_info.csv", f"{c_folder}/mutationMatrix.csv", f"{c_folder}")
                except Exception as e:
                     print(e)
                     with open("failed_projects/log.txt", "a") as f:
                         f.write(str(e))
                     shutil.move(c_folder, "failed_projects/")
        
        # copy original class
        for p in os.listdir(data_folder):
            for c in os.listdir( os.path.join(data_folder, p) ):
                c_folder = os.path.join(data_folder, p, c)
                #print(c_folder)
                if os.path.isfile(c_folder):
                    continue
                copy_original_class(f"{c_folder}")
    
   

