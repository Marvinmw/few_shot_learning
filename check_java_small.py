javapath="downstream/java-small/javas"
claspath="downstream/java-small/geometric"

import os
import json
for p in os.listdir(claspath):
    print(p)
    class_f = os.path.join(claspath, p, "class_file.json")
    java_f = os.path.join(javapath, p)
    javaset = set()
    for jf in os.listdir(java_f):
        jf = jf.strip()
        if jf.endswith(".java"):
            javaset.add(jf)
    classdict = json.load(open(class_f))
    print(len(classdict))
    print(len(javaset))
    print("=====")
    classet = set()
    for c in classdict:
        classet.add(c)
    d1=list(javaset - classet)
    d2=list(classet-javaset)
    print(len(list(javaset - classet)))
    print(len(list(classet-javaset)))    
    joinset = javaset.intersection(classet)
    print(len(joinset))