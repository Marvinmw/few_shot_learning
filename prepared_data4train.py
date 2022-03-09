from matplotlib import collections
#from utils.mutantsdataset_siamese import MutantsDataset
from utils.mutantsdataset_relevance import MutantRelevanceDataset, balanced_oversample
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
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


best_f1 = 0
view_test_f1 = 0
criterion = nn.CrossEntropyLoss()
contrastive_loss = ContrastiveLoss()

def train(args, model, device, loader, optimizer, loader_val, loader_test, epoch, saved_model_path, earlystopping, scheduler, dataset_list):
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
       
        if args.loss == "both":
            loss =   ( criterion( pred, batch.y)  + contrastive_loss(  x_s, x_t,  1-batch.y) )/2
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
            test_res={}
            avg_test_f1 = 0
            avg_test_recall = 0
            avg_test_acc = 0
            avg_test_pre = 0
            avg_test_loss = 0
            
            testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = eval(args, model, device, loader_test)
            test_res = [ testloss, accuracy_test, precision_test, recall_test, f1_test, result_test  ]
            avg_test_f1 += f1_test
            avg_test_recall += recall_test
            avg_test_acc += accuracy_test
            avg_test_pre += precision_test
            avg_test_loss += testloss

            earlystopping(f1_val, model, performance={"val_f1":f1_val, "test_f1":f1_test,"epoch":epoch, "test":test_res, 
                                            "val":[evalloss, accuracy_val, precision_val, recall_val, f1_val, testloss, result]})
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



import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    for p in projects:
        #set up dataset dataset_path
        dataset_inmemory = MutantRelevanceDataset( args.dataset_path , dataname=args.dataset, project=p)
        dataset_list[p] = dataset_inmemory
    return dataset_list

def create_dataset(args, train_projects, dataset_list):
    train_dataset = [ ]
    valid_dataset = [ ]
    test_dataset = []

    for tp in train_projects:
            dataset_inmemory = dataset_list[tp] 
            split_dict = dataset_inmemory.split(reshuffle=False, splitting_ratio=-1, fixed_size=30) 
            dataset = dataset_inmemory.data
            train_dataset.extend( [ dataset[i] for i in split_dict["train"].tolist() ])
            valid_dataset.extend( [ dataset[i] for i in split_dict["valid"].tolist() ] )
            test_dataset.extend( [ dataset[i] for i in  split_dict["test"].tolist()] )
    y = [ d.y.item() for d in train_dataset ]
    train_stat = collections.Counter(y)
    train_dataset = balanced_oversample(train_dataset, y)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    val_y = [ d.y.item() for d in valid_dataset ]
    val_stat = collections.Counter( val_y )
    print(len(valid_dataset))
    print(len(test_dataset))
    loader_val = DataLoader( valid_dataset, batch_size=min(int(args.batch_size/2), len(valid_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    loader_test = DataLoader( test_dataset, batch_size=min(int(args.batch_size/2),len(test_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t']  )
    test_y = [ d.y.item() for d in test_dataset ]
    test_stat = collections.Counter( test_y )

    return loader, loader_val, loader_test, train_projects, {"train":train_stat, "val":val_stat, "test":test_stat}

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
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
                projects[p].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    elif len(args.projects) == 1:
        for p in args.projects:
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
                n=os.path.basename(pf)
                projects[n].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    return projects, name


def train_mode(args):
   
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    
    projects, namelist = projects_dict(args)
    
    orgsavedpat = args.saved_model_path

    dataset_list = fecth_datalist(args, namelist)

   
    print(args.saved_model_path)
    

    loader, loader_val,  loader_test, train_projects, stat = create_dataset(args, train_projects, dataset_list)
    json.dump(train_projects, open(os.path.join(args.saved_model_path, "train_projects.json"), "w")  )
    #json.dump(remaining_projects, open(os.path.join(args.saved_model_path, "remaining_projects.json"), "w")  )
    json.dump(stat, open(os.path.join(args.saved_model_path, "stat.json"), "w")  , indent=6)
    num_class = args.num_class

   


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
    parser.add_argument('--dataset_path', type=str, default = 'dataset/relevant/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained_models/context/gat/model_0', help='filename to read the model (if there is any)')
    parser.add_argument('--saved_model_file', type=str, default= "-1", )
    #parser.add_argument('--pre', type=str, default= "yes", help="warm up training for this task")
    #parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'results/mutants_relevance/context', help='filename to output the pre-trained model')
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
    parser.add_argument('--lazy', type=str, default="no",
                        help='save model')
    parser.add_argument("--projects", nargs="+", default=["Mockito"])
    parser.add_argument("--loss", type=str, default="CT", help='[both, CT, CE]')
    parser.add_argument('--seed', type = int, default =1234)
    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    set_seed(args)
    train_mode(args)

    


if __name__ == "__main__":
    main()

