import os
import collections
import json
import re
def collect_mutants_type(folder_dir):
    mutants={}
    op_list = []
    op1_list = []
    op2_list = []
    for p in os.listdir(folder_dir):
        log_file = os.path.join(folder_dir, p, "mutants.log")
        with open(log_file) as f:
            for line in f.readlines():
                #print(line)
                line = f'{line}'
                line = re.sub(r'\\"', "#",line)
                #print(line)
                line = re.sub(r'""', "#",line)
                #print(line)
                line = re.sub(r'"(.+?)"', "#",line)
                #print(line)
                line = re.sub(r"':'", "#",line)
                #print(line)
                parts = line.split(":")
                #print(parts)
                mid = parts[0].strip()
                op  = parts[1].strip()
                op1 = parts[2].strip()
                op11 = re.sub(r'\(.+?\)', "",  op1)
                op11 = re.sub(r'#', "A", op11)
                if op1 not in ["0", "POS", "1", "NEG"] and re.match(r'^\w+', op11):
                    op11="METHOD_CALL"
                op2 = re.sub(r'\(.+?\)', "", parts[3].strip())
                mutants[f"{p}_{mid}"] = ( op, op1, op11, op2 )
                op_list.append(op)
                op1_list.append(op11)
                op2_list.append(op2)
    s=collections.Counter(op_list)
    print(f"Type Number {len(s)}")
    #s1=collections.Counter(op1_list)
    s2=collections.Counter(op2_list+op1_list)
    operand_id_mapping = {op:i+1 for i,op in enumerate(list(s2.keys()))}
    print(f"Operand Number {len(s2)}")
    sorted_s = sorted(s.items(), key= lambda x: x[1], reverse=True)
    #sorted_s1 = sorted(s1.items(), key= lambda x: x[1], reverse=True)
    sorted_s2 = sorted(s2.items(), key= lambda x: x[1], reverse=True)
    json.dump(sorted_s, open("sorted_operator.json", "w"), indent =6 )
    #json.dump(sorted_s1, open("sorted_operand1.json", "w"), indent =6 )
    json.dump(sorted_s2, open("sorted_operand.json", "w"), indent =6 )
    json.dump(mutants, open("mutants_type.json", "w"), indent = 6 )
    json.dump(operand_id_mapping, open("operand_id_mapping.json", "w"), indent = 6 )

if __name__ == "__main__":
   collect_mutants_type("defects4j/res_small")





