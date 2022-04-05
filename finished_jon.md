1. relevance
    a. few shot finetune using pretrained models,          few_shot_relevance_fine_tune_yes    Done
    b. few shot without finetune using pretrained models   few_shot_relevance_fine_tune_no     Done
    c. supervised relevance  without pretrained models     supervised_relevance                
    d. supervised relevance  with pretrained models        supervised_relevance_transferweights 
    e. 

2. killable
    a. few shot without finetune using pretrained models   few_shot_killed_fine_tune_no         Done             killed_k_fold_few_shot
    b. few shot finetune using pretrained models           few_shot_killed_fine_tune_yes        Done             killed_k_fold_few_shot
    c. few shot killing subsuming mutants   finetune no    few_shot_killed_subsuming_fine_tune_no  Done          killed_k_fold_few_shot_subsuming   
    d. few shot killing subsuming mutants   finetune yes   few_shot_killed_subsuming_fine_tune_yes  need_eval    killed_k_fold_few_shot_subsuming
    e. supervised_killed_subsuming   using pretrained      supervised_killed_subsuming_transferweights Done      killed_k_fold_transfer_subsuming
    f. supervised_killed    using pretrained               supervised_killed_transferweights           Done      killed_k_fold_transfer
