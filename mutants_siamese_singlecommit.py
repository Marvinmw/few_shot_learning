from cmath import inf
from matplotlib import collections
from utils.mutantsdataset_siamese import MutantsDataset, balanced_oversample
import argparse
import json
from torch_geometric.data import DataLoader
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from utils.model import  GNN_encoder
from utils.tools import performance, TokenIns
from utils.pytorchtools import EarlyStopping
from utils.AverageMeter import AverageMeter
from utils.probing_classifier import PredictionLinearModelFineTune
from utils.ContrastiveLoss import ContrastiveLoss 
import collections
import random
import gc
try:
    from transformers import get_linear_schedule_with_warmup as linear_schedule
except:
    print("import WarmupLinearSchedule instead of get_linear_schedule_with_warmup")
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
    def linear_schedule(optimizer, num_warmup_steps=100, num_training_steps=100):
        return get_linear_schedule_with_warmup(optimizer, warmup_steps=num_warmup_steps,
                                                    t_total=num_training_steps)
                                                    
best_f1 = 0
view_test_f1 = 0
criterion = nn.CrossEntropyLoss()
contrastive_loss = ContrastiveLoss()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args, model, device, loader, optimizer, loader_val, test_loader_dic, epoch, saved_model_path, earlystopping, scheduler, dataset_list):
    global best_f1
    global view_test_f1
    model.train()
    trainloss = AverageMeter()
    res = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()      
        pred,  x_s, x_t = model(batch)    
        # if args.num_class == 2:
        #     batch.y[ batch.y != 0 ] = 1
        #     #batch.y = 1 -  batch.y
        # else:
        #     batch.y[ batch.y == 4 ] = 1
        if args.loss == "both":
            loss =  ( criterion( pred, batch.y) + contrastive_loss( x_s, x_t, 1 - batch.y)    )/2
        elif args.loss == "CE":
            loss =  criterion( pred, batch.y) 
        elif args.loss == "CT":
            loss = contrastive_loss( x_s, x_t, 1 - batch.y)
        else:
            assert False
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if args.warmup_schedule:
            scheduler.step()  
        trainloss.update(loss.item())
        y_true.extend( batch.y.detach().cpu())
        _, predicted_labels = torch.max( pred, dim=1 )
        y_pred.extend(predicted_labels.detach().cpu())
        if step%args.save_steps == 0 :
            print("Evaluation ")
            model.eval()
            evalloss, accuracy_val, precision_val, recall_val, f1_val, result = eval(args, model, device, loader_val)
            test_res_sum ={}
            avg_test_f1 = 0
            avg_test_recall = 0
            avg_test_acc = 0
            avg_test_pre = 0
            avg_test_loss = 0
            for tproject in test_loader_dic:
                testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = eval(args, model, device, test_loader_dic[tproject])
                test_res = [ testloss, accuracy_test, precision_test, recall_test, f1_test, result_test  ]
                test_res_sum[tproject] = test_res
                avg_test_f1 += f1_test
                avg_test_recall += recall_test
                avg_test_acc += accuracy_test
                avg_test_pre += precision_test
                avg_test_loss += testloss

            avg_test_f1 = avg_test_f1/len(test_loader_dic)
            avg_test_recall = avg_test_recall/len(test_loader_dic)
            avg_test_acc = avg_test_acc/len(test_loader_dic)
            avg_test_pre = avg_test_pre/len(test_loader_dic)
            avg_test_loss = avg_test_loss/len(test_loader_dic)

            earlystopping(f1_val, model, performance={"val_f1":f1_val, "test_f1":avg_test_f1,"epoch":epoch, "test":test_res_sum, 
                                            "val":[evalloss, accuracy_val, precision_val, recall_val, f1_val, avg_test_loss, result]})
            model.train()
            print(f"Best Test {view_test_f1}")
            print(f"\nEpoch {step}/{epoch}, Valid, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
            print(f"\nEpoch {step}/{epoch}, Test, Eval Loss {avg_test_loss}, Accuracy {avg_test_acc}, Precision {avg_test_pre}, Recall {avg_test_recall}, F1 {avg_test_f1}"  )
            if f1_val > best_f1 :
                best_f1 = f1_val
                view_test_f1 = avg_test_f1
                if earlystopping.save_model:
                    torch.save(model.state_dict(), os.path.join(saved_model_path,  f"best_epoch{epoch}_.pth"))
            res.append([accuracy_val, precision_val, recall_val, f1_val])
    model.eval()
    print("Evaluation ")
    epcoh_res = []
    accuracy_train, precision_train, recall_train, f1_train, = performance(y_true, y_pred, average="macro")
    print(f"\nEpoch {epoch}, Train,  Loss {trainloss.avg}, Accuracy {accuracy_train}, Precision {precision_train}, Recall {recall_train}, F1 {f1_train}"  )
    epcoh_res.extend( [accuracy_train, precision_train, recall_train, f1_train ] )
    evalloss, accuracy_val, precision_val, recall_val, f1_val,_ = eval(args, model, device, loader_val)
    print(f"\nEpoch {epoch}, Valid,  Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    epcoh_res.extend( [ evalloss, accuracy_val, precision_val, recall_val, f1_val ] )
    return epcoh_res, res, f1_val

def eval(args, model, device, loader):
    y_true = []
    y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs,x_s, x_t = model(batch)
            # if args.num_class == 2:
            #     batch.y[ batch.y!= 0 ] = 1
            # else:
            #     batch.y[ batch.y == 4 ] = 1
            #loss = criterion( outputs, batch.y)
            if args.loss == "both":
                loss =   criterion( outputs, batch.y)*model.beta + contrastive_loss( x_s, x_t, 1 - batch.y)   *model.alpha
            elif args.loss == "CE":
                loss =  criterion( outputs, batch.y) 
            elif args.loss == "CT":
                loss = contrastive_loss( x_s, x_t, 1 - batch.y)
            else:
                assert False
            evalloss.update( loss.item() )         
        y_true.append(batch.y.cpu())
        _, predicted_label = torch.max( outputs, dim=1 )
        y_prediction.append(predicted_label.cpu())
      
    y_true = torch.cat(y_true, dim = 0)
    y_prediction = torch.cat(y_prediction, dim = 0)
    accuracy, precision, recall, f1 = performance( y_true,y_prediction, average="binary")
    accuracy_macro, precision_macro, recall_macro, f1_macro = performance( y_true,y_prediction, average="macro")
    accuracy_weighted, precision_weighted, recall_weighted, f1_weighted = performance( y_true,y_prediction, average="weighted")
    accuracy_micro, precision_micro, recall_micro, f1_micro = performance( y_true,y_prediction, average="micro") 
    result = {"eval_accuracy":accuracy, "eval_precision":precision, "eval_recall":recall,"eval_f1": f1, "macro":[accuracy_macro, precision_macro, recall_macro, f1_macro],
    "weighted":[accuracy_weighted, precision_weighted, recall_weighted, f1_weighted], "micro":[accuracy_micro, precision_micro, recall_micro, f1_micro]}
    return evalloss.avg, accuracy, precision, recall, f1, result

import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    
    for p in projects:
        #set up dataset dataset_path
        dataset_inmemory = MutantsDataset( args.dataset_path , dataname=args.dataset, project=p)
        dataset_list[p] = dataset_inmemory
        
    return dataset_list

def create_dataset(args, dataset_list):
    train_dataset = [ ]
    valid_dataset = [ ]
    
    train_project = None
    minimum_size = inf
    for tp in dataset_list:
            dataset_inmemory = dataset_list[tp] 
            if args.loss == "CE":
                split_dict = dataset_inmemory.split(reshuffle=False, splitting_ratio=0.1) 
            else:
                split_dict = dataset_inmemory.split(reshuffle=False, splitting_ratio=0.25) 
            if args.task == "killable":
                dataset_inmemory.keep_label(zeros_label=[0])
            elif args.task == "fail":
                dataset_inmemory.keep_label(zeros_label=[0,2,3])
            elif args.task == "exc":
                dataset_inmemory.keep_label(zeros_label=[0,1,3,4])
            elif args.task == "time":
                dataset_inmemory.keep_label(zeros_label=[0,1,2,4])
            else:
                assert False, f"Wrong task, {args.task}"
            dataset = dataset_inmemory.data
            print(f"{tp} {len(dataset)}")
            if minimum_size > len(dataset):
                train_project = tp
                minimum_size = len(dataset)
            #train_dataset.extend( [ dataset[i] for i in split_dict["train"].tolist() ])
            #valid_dataset.extend( [ dataset[i] for i in split_dict["valid"].tolist() ] )
            #test_dataset.extend( [ dataset[i] for i in  split_dict["test"].tolist()] )
    
    train_dataset_inmemory =   dataset_list[train_project]     
    train_split_dict = train_dataset_inmemory.get_split_index(splitting_ratio=0.25)  
    train_data = train_dataset_inmemory.data
    #print(train_split_dict["train"].tolist())
    train_dataset.extend( [ train_data[i] for i in train_split_dict["train"].tolist() ])
    valid_dataset.extend( [ train_data[i] for i in train_split_dict["valid"].tolist()+train_split_dict["test"].tolist() ] )
    #test_dataset.extend( [ train_data[i] for i in  train_split_dict["test"].tolist()] )

    y = [ d.y.item() for d in train_dataset ]
    train_stat = collections.Counter(y)
    train_dataset = balanced_oversample(train_dataset, y)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    val_y = [ d.y.item() for d in valid_dataset ]
    val_stat = collections.Counter( val_y )
    print(len(valid_dataset))
    #print(len(test_dataset))
    loader_val = DataLoader( valid_dataset, batch_size=min(int(args.batch_size/2), len(valid_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    test_loader_dic = {}
    test_y = []
    for tp in dataset_list:
        if tp != train_project:
            dataset_inmemory = dataset_list[tp] 
           # split_dict = dataset_inmemory.split(reshuffle=False, splitting_ratio=0.1) 
            dataset = dataset_inmemory.data
            test_loader_dic[tp] = DataLoader( dataset, batch_size=min(int(args.batch_size/2),len(dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t']  )
            test_y = test_y + [ d.y.item() for d in dataset ]
    test_stat = collections.Counter( test_y )

    return loader, loader_val, test_loader_dic, train_project, {"train":train_stat, "val":val_stat, "test":test_stat}

def fetch_dataset(args, dataset_inmemory):
    #dataset_inmemory = MutantsDataset( args.dataset_path , dataname=args.dataset, project=name)
    #split_dict = dataset_inmemory.split(reshuffle=False) 
    dataset = dataset_inmemory.data
    loader_test = DataLoader( dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    return loader_test

def projects_dict(args):
    projects = collections.defaultdict(list)
    name=[]
    if len(args.projects) > 1:
        for p in args.projects:
            for pf in glob.glob(f"{args.dataset_path}/{p}_*_fixed"):
                projects[p].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    elif len(args.projects) == 1:
        for p in args.projects:
            for pf in glob.glob(f"{args.dataset_path}/{p}_*_fixed"):
                print(pf)
                n=os.path.basename(pf)
                projects[n].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    return projects, name


def train_mode(args):
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    
    projects, namelist = projects_dict(args)
    
   
    #klist = list( projects.keys() )
    # for pp in klist:
    #             train_projects.extend(projects[pp])
    orgsavedpat=args.saved_model_path

    dataset_list = fecth_datalist(args, namelist)

    args.saved_model_path = f"{orgsavedpat}"
    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    print(args.saved_model_path)
    

    loader, loader_val,  test_loader_dic, train_projects, stat = create_dataset(args, dataset_list)
    json.dump([train_projects], open(os.path.join(args.saved_model_path, "train_projects.json"), "w")  )
    #json.dump(remaining_projects, open(os.path.join(args.saved_model_path, "remaining_projects.json"), "w")  )
    json.dump(stat, open(os.path.join(args.saved_model_path, "stat.json"), "w")  , indent=6)
    num_class = args.num_class

    #set up model
    tokenizer_word2vec = TokenIns( 
            word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
            tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
        )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                            graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                            bidrection=args.bidirection, task="mutants", repWay=args.repWay)
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Trainable Parameters Encoder {pytorch_total_params}\n")
    
    encoder.gnn.embedding.fine_tune_embeddings(True)
    if not args.input_model_file == "-1":
            encoder.gnn.embedding.init_embeddings(embeddings)
            print(f"Load Pretraning model {args.input_model_file}")
            encoder.from_pretrained(args.input_model_file + ".pth", device)
  
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model = PredictionLinearModelFineTune(600, num_class,encoder,False if args.mutant_type == "no" else True,args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters Model {pytorch_total_params}\n")
    if not args.saved_model_file == "-1":
        model.load_state_dict( torch.load(args.saved_model_file, map_location="cpu"), strict=False )
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.decay ) 
    args.max_steps=args.epochs*len( loader)
    args.save_steps=max(1, len( loader)//10)
    args.warmup_steps=args.max_steps//5
    scheduler = linear_schedule(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=args.max_steps)
    args.warmup_schedule = False if args.warmup_schedule == "no" else True
    save_model = False if args.lazy == "yes" else True
    f0 = open(args.log_file, "w")
    f1 = open( args.log_file+"_epoch" , "w")
    f = csv.writer( f0 )
    ef = csv.writer( f1 )
    f.writerow(["Accuracy", "Precsion", "Recall", "F1"])
    ef.writerow(["Accuracy", "Precsion", "Recall", "F1", "Val Loss","Val Accuracy", "Val Precsion", "Val Recall", "Val F1", "Test Loss",
                        "Test Accuracy", "Test Precsion", "Test Recall", "Test F1"])
    epoch_res = [ ]
    if args.loss == "CT":
        earlystopping = EarlyStopping(monitor="loss", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    else:
        earlystopping = EarlyStopping(monitor="f1", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    for epoch in range(1, args.epochs+1):
        print(" ====epoch === " + str(epoch))
        performance_model, evalstepres, _  = train(args, model, device, loader, optimizer, loader_val, test_loader_dic,epoch, args.saved_model_path, earlystopping, scheduler, dataset_list)
        epoch_res.append( performance_model )
        for r in evalstepres:
            f.writerow(r)
    f0.close()
    for r in epoch_res:
        ef.writerow(r)
    f1.close()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', default=False) 
    parser.add_argument('--remove_gnn_attention', dest='remove_gnn_attention', action='store_true', default=False) 
    parser.add_argument('--test', type=str, dest='test', default="") 
    parser.add_argument('--subword_embedding', type=str, default="lstm",
                        help='embed  (bag, lstmbag, gru, lstm, attention, selfattention)')
    parser.add_argument('--bidirection', dest='bidirection', action='store_true', default=True) 
    parser.add_argument('--lstm_emb_dim', type=int, default=150,
                        help='lstm embedding dimensions (default: 512)')
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="sum",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gat")
    parser.add_argument('--repWay', type=str, default="append", help='seq, append, graph, alpha')
    parser.add_argument('--nonodeembedding', dest='nonodeembedding', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/mutants_small/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained_models/context/gat/model_0', help='filename to read the model (if there is any)')
    parser.add_argument('--saved_model_file', type=str, default= "-1", )
    parser.add_argument('--task', type=str, default= "killable", )
    #parser.add_argument('--pre', type=str, default= "yes", help="warm up training for this task")
    parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'results/mutants_siamese/context', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--num_class', type = int, default =2, help='num_class')
    parser.add_argument('--dropratio', type = float, default =0.25, help='drop_ratio')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--warmup_schedule', type=str, default="no",
                        help='warmup')
    parser.add_argument('--mutant_type', type=str, default="no",
                        help='mutantype')
    parser.add_argument('--lazy', type=str, default="yes",
                        help='save model')
    parser.add_argument("--projects", nargs="+", default=["JxPath"])
    parser.add_argument("--loss", type=str, default="both", help='[both, CT, CE]')
    parser.add_argument('--seed', type = int, default =1234)
    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    set_seed(args)
    train_mode(args)

    


if __name__ == "__main__":
    main()

