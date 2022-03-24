import json
import glob
import collections
from unittest import result
import pandas as pd
import os
from scipy.stats import mannwhitneyu, ttest_rel
def read_random(p):
    rpair = json.load( open( os.path.join(p, "random_pair.json") ) )
    rsingle = json.load( open( os.path.join(p, "random_single.json") ) )
    return rpair, rsingle

def read_fewshot(p):
    rpair = json.load( open( os.path.join(p, "few_shot_test_pair.json") ) )
    rsingle = json.load( open( os.path.join(p, "few_shot_test_single.json") ) )
    return rpair, rsingle

def read_ce(p):
    rpair = json.load( open( os.path.join(p, "ranking_eval_pair.json") ) )
    rsingle = json.load( open( os.path.join(p, "ranking_eval_single.json") ) )
    return rpair, rsingle

def compare(project_name, t="pr_area"):
    result = {}
    random_yes_dir = f"results/random_prior/yes/mutants_{project_name}/context/"
    random_yes_pair, random_yes_single = read_random( random_yes_dir )
    random_no_dir = f"results/random_prior/no/mutants_{project_name}/context/"
    random_no_pair, random_no_single = read_random( random_no_dir )

    few_shot_relevance_no_dir = f"results/few_shot_relevance_fine_tune_no/mutants_{project_name}/context/gat"
    few_no_pair, few_no_single = read_fewshot( few_shot_relevance_no_dir )

    few_shot_relevance_yes_dir = f"results/few_shot_relevance_fine_tune_yes/mutants_{project_name}/context/gat"
    supervised = f"results/supervised_relevance_transferweights/CE/mutants_{project_name}/context/gat"

    # compare two random algorithms, few shot without finetune
    namelist = list(random_yes_pair.keys())
    random_yes_pair_list = []
    random_no_pair_list = []
    random_yes_single_list = []
    random_no_single_list = []
    few_no_pair_list = []
    few_no_single_list = []
    for n in namelist:
         if n in few_no_pair and n in random_yes_pair and n in few_no_single:
            random_yes_pair_list.append( random_yes_pair[n][t] )
            random_no_pair_list.append( random_no_pair[n][t] )
            random_yes_single_list.append( random_yes_single[n][t] )
            random_no_single_list.append( random_no_single[n][t] )
            few_no_pair_list.append( few_no_pair[n][t] )
            few_no_single_list.append( few_no_single[n][t] )
    U, p = mannwhitneyu(random_yes_pair_list, random_no_pair_list, alternative="greater" )
    print(f"Pair random with prior and withoug prior, {len(random_yes_pair_list)}, U-Test {U} {p}")
    result["random_compare"] = p
    U, p = mannwhitneyu(random_yes_single_list, random_no_single_list, alternative="greater" )
    print(f"Single random with prior and withoug prior,{len(random_yes_single_list)}, U-Test {U} {p}")

    U, p = ttest_rel(few_no_pair_list, random_yes_pair_list, alternative="greater" )
    print(f"Pair Few-no random , {len(few_no_pair_list)},T-Test {U} {p}")
    result["few_shot_random_compare_t_test"] = p
    U, p = ttest_rel(few_no_single_list, random_yes_single_list, alternative="greater" )
    print(f"Single Few-no random , {len(few_no_single_list)},T-Test {U} {p}")
    

    U, p = mannwhitneyu(few_no_pair_list, random_yes_pair_list, alternative="greater" )
    print(f"Pair Few-no random , {len(few_no_pair_list)},U-Test {U} {p}")
    result["few_shot_random_compare_u_test"] = p
    U, p = mannwhitneyu(few_no_single_list, random_yes_single_list, alternative="greater" )
    print(f"Single Few-no random , {len(few_no_single_list)},U-Test {U} {p}")

    # ss, pt = ttest_rel(random_yes, random_no, equal_var=True)
    # print(f"random with prior and withoug prior Paired-T-Test {ss} {pt}")

    # compare fine-tune-few-shot
    p_values_compare_list = [ ]
    larger_ratio_list = []
    for n in namelist:
        p = os.path.join( few_shot_relevance_yes_dir, f"{n}_fold")
        try:
            few_yes_pair, few_yes_single = read_fewshot(p)
        except:
            continue
        random_yes_pair_list = []
        random_yes_single_list = []
        few_yes_pair_list = []
        few_yes_single_list = []
        for j in namelist:
            if j == n: 
                continue
            if j in few_yes_pair and j in random_yes_pair and j in few_yes_single:
                random_yes_pair_list.append( random_yes_pair[j][t] )
                random_yes_single_list.append( random_yes_single[j][t] )
                few_yes_pair_list.append( few_yes_pair[j][t] )
                few_yes_single_list.append( few_yes_single[j][t] )
            
        U, p = mannwhitneyu(few_yes_pair_list, random_yes_pair_list, alternative="greater" )
        c=0
        for i in range(len(few_yes_pair_list)):
            if few_yes_pair_list[i] > random_yes_pair_list[i]:
                c = c + 1
        larger_ratio = c/len(few_yes_pair_list)
        larger_ratio_list.append( larger_ratio )
        print(f"Pair Few-no random , {len(few_yes_pair_list)},U-Test {U} {p}")
        p_values_compare_list.append( p )

        U, p = mannwhitneyu(few_yes_single_list, random_yes_single_list, alternative="greater" )
        print(f"Single Few-no random , {len(few_yes_single_list)},U-Test {U} {p}")
    
    result["few_shot_yes_random"] = p_values_compare_list
    result["large_ratio_random"] = larger_ratio_list

    # compare fine-tune-few-shot-CE
    p_values_compare_list = [ ]
    larger_ratio_list = []
    for n in namelist:
        p = os.path.join( few_shot_relevance_yes_dir, f"{n}_fold")
        sp = os.path.join( supervised, f"{n}_fold")
        try:
            few_yes_pair, few_yes_single = read_fewshot(p)
            ce_pair, ce_single = read_ce(sp)
        except:
            continue
        ce_yes_pair_list = []
        ce_yes_single_list = []
        few_yes_pair_list = []
        few_yes_single_list = []
        for j in namelist:
            if j == n: 
                continue
            if j in few_yes_pair and j in ce_pair and j in few_yes_single:
                ce_yes_pair_list.append( ce_pair[j][t] )
                ce_yes_single_list.append( ce_single[j][t] )
                few_yes_pair_list.append( few_yes_pair[j][t] )
                few_yes_single_list.append( few_yes_single[j][t] )
            
        U, p = mannwhitneyu(few_yes_pair_list, ce_yes_pair_list, alternative="greater" )
        c=0
        for i in range(len(few_yes_pair_list)):
            if few_yes_pair_list[i] > ce_yes_pair_list[i]:
                c = c + 1
        larger_ratio = c/len(few_yes_pair_list)
        larger_ratio_list.append( larger_ratio )
        print(f"Pair Few-no random , {len(few_yes_pair_list)},U-Test {U} {p}")
        p_values_compare_list.append( p )

        U, p = mannwhitneyu(few_yes_single_list, ce_yes_single_list, alternative="greater" )
        print(f"Single Few-no random , {len(few_yes_single_list)},U-Test {U} {p}")
    
    result["few_shot_yes_ce"] = p_values_compare_list
    result["large_ratio_ce"] = p_values_compare_list
    return result

def box_plot(data, xlabel, name):
    import matplotlib.pyplot as plt
     
    fig = plt.figure(figsize =(10, 7))
    
    # # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
    
    # Creating plot
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4], xlabel)
    plt.title(name)
    # show plot
    plt.show()

if __name__ == "__main__":
    print("--- collections ---")
    res_collections = compare("collections")
    print("--- lang ---")
    res_lang = compare("lang")
    print("--- text ---")
    res_text = compare("text")
    print("--- io ---")
    res_io = compare("io")
    print("--- csv ---")
    res_csv = compare("csv")

    json.dump( res_collections, open("collections.json", "w"), indent=6  )
    json.dump( res_lang, open("lang.json", "w"), indent=6  )
    json.dump( res_text, open("text.json", "w"), indent=6  )
    json.dump( res_io, open("io.json", "w"), indent=6  )
    json.dump( res_csv, open("csv.json", "w"), indent=6  )

    box_plot([res_collections["few_shot_yes_ce"], res_lang["few_shot_yes_ce"], res_text["few_shot_yes_ce"],
        res_csv["few_shot_yes_ce"]  ], ["collections", "lang", "text", "csv"],"Few-Shot VS Supervised, P-Value")

    box_plot([res_collections["few_shot_yes_random"], res_lang["few_shot_yes_random"], res_text["few_shot_yes_random"],
        res_csv["few_shot_yes_random"]  ], ["collections", "lang", "text", "csv"], "Few-Shot VS Random, P-Value")

    
    box_plot([res_collections["large_ratio_ce"], res_lang["large_ratio_ce"], res_text["large_ratio_ce"],
        res_csv["large_ratio_ce"]  ], ["collections", "lang", "text", "csv"], "Few-Shot VS Random, Greater Times")
    
    
