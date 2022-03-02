from argparse import ArgumentParser
import json
import collections
import pandas as pd
import glob
import os
from pathlib import Path
import shutil

def rename_folder(mutatns_folder):
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

def replace_name(mutatns_folder, mutants_info, outputfolder):
    df = pd.read_csv(mutants_info, index_col=["lineNumber","index", "block", "mutator"])
    df.rename({"MutantID":"ind"}, inplace=True)
    df.rename({"lineNumber":"lineNumbers"}, inplace=True)
    df.rename({"index":"indexes"}, inplace=True)
    df.rename({"block":"blocks"}, inplace=True)
    df.rename({"mutator":"mutators"}, inplace=True)
    mutant_info = collections.defaultdict()
    
    for f in glob.glob(f"{mutatns_folder}/**/details.txt" , recursive=True):    
        mid = os.path.basename( os.path.dirname(f) )
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
            
            #print(f)
            if (int(info["lineNumbers"]), int(info["indexes"]), int(info["blocks"]), info["mutators"] ) in df.index:
                r = df.loc[(int(info["lineNumbers"]), int(info["indexes"]), int(info["blocks"]), info["mutators"] ), :]
                if r["Relevant"] + r["Not_relevant"] + r["On_Change"]:
                    info["label"] = 1 if r["On_Change"] + r["Relevant"] > 0 else 0
                    info["methodDescription"] = r["methodDescription"]
                    info["mutatedMethod"] = r["mutatedMethod"]
                    info["mutatedClass"] = r["mutatedClass"]
                    info["On_Change"] = str(r["On_Change"])
                    mutant_info[mid] = info

    for f in glob.glob(f"{mutatns_folder}/**/**.class" , recursive=True):        
        p = Path(f)
        fname = os.path.basename( p )
        name = ".".join( fname.split(".")[-2:] )
        p.rename(Path(p.parent, f"{name}"))            
                
    json.dump(mutant_info, open(f"{outputfolder}/mutants_info.json", "w"), indent=6 )
            
        
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("-i", "--inputfile", dest="inputfile")
#     parser.add_argument("-o", "--outputfile", dest="outputfile")
#     args = parser.parse_args()
#     replace_name(args.inputfile, args.outputfile)

replace_name("fom_export", "fom_export/mutants_info.csv", "fom_dot")