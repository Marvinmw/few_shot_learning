from argparse import ArgumentParser
from itertools import count
import json
import collections
import pandas as pd
import glob
import os
from pathlib import Path
import shutil

        
if __name__ == '__main__':

    #rename folder
    data_folder = "relevance_java_bytecode"

    
    # extract graph
    counter=1
    
    for p in os.listdir(data_folder):
            for c in os.listdir( os.path.join(data_folder, p) ):
                c_folder = os.path.join(data_folder, p, c)
                if os.path.isdir(c_folder):
                    with open(f"job_{counter//30}.sh", "a") as f:
                        f.write("#!/bin/bash \n")
                       # f.write(f"java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i {c_folder}/fom_export/ -o relevance_java_dot_byteinfo/{p}/{c}/mutants \n")
                       # f.write(f"java -jar GraphExtractor/target/extracterGraph-1.0-SNAPSHOT.jar -g  -i {c_folder}/original/ -o relevance_java_dot_byteinfo/{p}/{c}/original \n")
                        f.write(f"cp {c_folder}/mutants_info.json relevance_java_dot_byteinfo/{p}/{c}/ \n")
                        f.write(f"cp {c_folder}/interaction_pairs.csv relevance_java_dot_byteinfo/{p}/{c}/ \n")
                    counter = counter + 1
                    print(counter//30)