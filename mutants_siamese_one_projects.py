import sys
# setting path
sys.path.append('../')
from utils.mutantsdataset import MutantKilledDataset, MutantRelevanceDataset, balanced_oversample
import argparse
import json
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
from utils.model import  GNN_encoder
from utils.tools import performance, TokenIns, get_logger
from utils.pytorchtools import EarlyStopping
from utils.AverageMeter import AverageMeter
from utils.classifier import PredictionLinearModelFineTune
from utils.ContrastiveLoss import ContrastiveLoss, SelfContrastiveLoss
import collections
import random
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
temprature = 0.3  # temprature for contrastive loss
lam = 0.9  # lambda for loss
self_contrastive_loss = SelfContrastiveLoss(temprature)

def train(args, model, device, loader, optimizer, scheduler):
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
        if args.num_class == 2:
            y = batch.by
        else:
            y = batch.my
       
        if args.loss == "both":
            loss =  lam * criterion( pred, y) + contrastive_loss( x_s, x_t, 1 - y)  * (1 - lam)
        elif args.loss == "CE":
            loss =  criterion( pred, y) 
        elif args.loss == "CL":
            loss = contrastive_loss( x_s, x_t, 1 - y)
        elif args.loss == "SCL":
            cross_loss =  criterion( pred, y) 
            contrastive_l = self_contrastive_loss( torch.cat( (x_s, x_t),  dim=1).cpu().detach().numpy(), y.cpu().detach().numpy() ) 
            loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
        else:
            assert False, f"Wrong loss name {args.loss}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if args.warmup_schedule:
            scheduler.step()  
        trainloss.update(loss.item())
        y_true.extend( y.detach().cpu())
        _, predicted_labels = torch.max( pred, dim=1 )
        y_pred.extend(predicted_labels.detach().cpu())
        #if step%args.save_steps == 0 and args.evalutaionInEpoch :
        #    print("Evaluation ")
        #    model.eval()
        #    evalloss, accuracy_val, precision_val, recall_val, f1_val, result = eval(args, model, device, loader_val)
            # test_res={}
            # avg_test_f1 = 0
            # avg_test_recall = 0
            # avg_test_acc = 0
            # avg_test_pre = 0
            # avg_test_loss = 0
            
            # testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = eval(args, model, device, loader_test)
            # test_res = [ testloss, accuracy_test, precision_test, recall_test, f1_test, result_test  ]
            # avg_test_f1 += f1_test
            # avg_test_recall += recall_test
            # avg_test_acc += accuracy_test
            # avg_test_pre += precision_test
            # avg_test_loss += testloss

            # earlystopping(f1_val, model, performance={"val_f1":f1_val, "test_f1":f1_test,"epoch":epoch, "test":test_res, 
            #                                 "val":[evalloss, accuracy_val, precision_val, recall_val, f1_val, testloss, result]})
        #    earlystopping(f1_val, model, performance={"val_f1":f1_val, "epoch":epoch, 
        #                                    "val":[evalloss, accuracy_val, precision_val, recall_val, f1_val, result]})
        #    model.train()
        #    print(f"Best Test {view_test_f1}")
        #    print(f"\nEpoch {step}/{epoch}, Valid, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
            #print(f"\nEpoch {step}/{epoch}, Test, Eval Loss {avg_test_loss}, Accuracy {avg_test_acc}, Precision {avg_test_pre}, Recall {avg_test_recall}, F1 {avg_test_f1}"  )
        #    if f1_val > best_f1 :
        #        best_f1 = f1_val
                #view_test_f1 = avg_test_f1
                # if earlystopping.save_model:
                #     torch.save(model.state_dict(), os.path.join(saved_model_path,  f"best_epoch{epoch}_.pth"))
        #    res.append([accuracy_val, precision_val, recall_val, f1_val])
    
    # print("Evaluation ")
    # epcoh_res = []
    # accuracy_train, precision_train, recall_train, f1_train, = performance(y_true, y_pred, average="binary")
    # print(f"\nEpoch {epoch}, Train,  Loss {trainloss.avg}, Accuracy {accuracy_train}, Precision {precision_train}, Recall {recall_train}, F1 {f1_train}"  )
    # epcoh_res.extend( [accuracy_train, precision_train, recall_train, f1_train ] )
    # evalloss, accuracy_val, precision_val, recall_val, f1_val,_ = eval(args, model, device, loader_val)
    # print(f"\nEpoch {epoch}, Valid,  Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    # epcoh_res.extend( [ evalloss, accuracy_val, precision_val, recall_val, f1_val ] )
    #return epcoh_res, res, f1_val
    return trainloss.avg

def evalutaion(args, model, device, loader_val, epoch, earlystopping ):
    global best_f1
    global view_test_f1
    model.eval()
    evalloss, accuracy_val, precision_val, recall_val, f1_val, result = eval(args, model, device, loader_val)
    earlystopping(f1_val, model, performance={"val_f1":f1_val, "epoch":epoch,
                                            "val":[evalloss, accuracy_val, precision_val, recall_val, f1_val, result]})
      
    #logger.info(f"Best Test {view_test_f1}")
    logger.info(f"Epoch {epoch}, Valid, Eval Loss {evalloss}, Accuracy {accuracy_val}, Precision {precision_val}, Recall {recall_val}, F1 {f1_val}"  )
    if f1_val > best_f1 :
            best_f1 = f1_val
    return evalloss, accuracy_val, precision_val, recall_val, f1_val, result

def build_model(args, device):
     #set up model
    tokenizer_word2vec = TokenIns( 
            word2vec_file=os.path.join(args.sub_token_path, args.emb_file),
            tokenizer_file=os.path.join(args.sub_token_path, "fun.model")
        )
    embeddings, word_emb_dim, vocab_size = tokenizer_word2vec.load_word2vec_embeddings()
    encoder = GNN_encoder(args.num_layer,vocab_size, word_emb_dim, args.lstm_emb_dim, args.num_class, JK = args.JK, drop_ratio = args.dropout_ratio, 
                            graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, subword_emb=args.subword_embedding,
                            bidrection=args.bidirection, task="mutants", repWay=args.repWay)
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters Encoder {pytorch_total_params}\n")
    
    encoder.gnn.embedding.fine_tune_embeddings(True)
    if not args.input_model_file == "-1":
            encoder.gnn.embedding.init_embeddings(embeddings)
            logger.info(f"Load Pretraning model {args.input_model_file}")
            encoder.from_pretrained(args.input_model_file + ".pth", device)
  
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model = PredictionLinearModelFineTune(600, args.num_class, encoder, args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters Model {pytorch_total_params}\n")
    model.load_state_dict( torch.load(os.path.join(args.saved_model_path, "saved_model.pt"), map_location="cpu") )
    model.to(device)
    return model

def test_eval(args, model, device, loader_test):
   
    model.eval()
    testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = eval(args, model, device, loader_test)
    logger.info(f" Test, Eval Loss {testloss}, Accuracy {accuracy_test}, Precision {precision_test}, Recall {recall_test}, F1 {f1_test}"  )
    return testloss, accuracy_test, precision_test, recall_test, f1_test, result_test

def eval(args, model, device, loader):
    y_true = []
    y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            outputs,x_s, x_t = model(batch)
            if args.num_class == 2:
                y = batch.by
            else:
                y = batch.my

            if args.loss == "both":
                loss =  lam * criterion( outputs, y) + contrastive_loss( x_s, x_t, 1 - y)  * (1 - lam)
            elif args.loss == "CE":
                loss =  criterion( outputs, y) 
            elif args.loss == "CL":
                loss = contrastive_loss( x_s, x_t, 1 - y)
            elif args.loss == "SCL":
                cross_loss =  criterion( outputs, y) 
                contrastive_l = self_contrastive_loss( torch.cat( (x_s, x_t),  dim=1).cpu().detach().numpy(), y.cpu().detach().numpy() ) 
                loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
            else:
                assert False, f"Wrong loss name {args.loss}"
            evalloss.update( loss.item() )         
        y_true.append(y.cpu())
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
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance":
            dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p, probability=0 )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list

def create_dataset(args, train_projects, dataset_list):
    train_dataset = [ ]
    valid_dataset = [ ]
   
    data = []
    for tp in train_projects:
            if args.task == "killed":
                dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{tp}" , dataname=args.dataset, project=tp )
            elif args.task == "relevance":
                dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{tp}" , dataname=args.dataset, project=tp, probability=0.8 )
            else:
                assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
            dataset = dataset_inmemory.data
            data.extend( dataset )
    #data=data[:2000] # for local debug
    #args.batch_size=64 # for local debug
    random.shuffle(data)
    val_size = int(len(data)*0.2)
    train_size = len(data) -val_size
    train_dataset = data[:train_size]
    valid_dataset = data[train_size : ]
   
    y = [ d.by.item() for d in train_dataset ]
    train_stat = collections.Counter(y)
    train_dataset = balanced_oversample(train_dataset, y)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    val_y = [ d.by.item() for d in valid_dataset ]
    val_stat = collections.Counter( val_y )
    print(len(valid_dataset))
    #print(len(test_dataset))
    loader_val = DataLoader( valid_dataset, batch_size=min(int(args.batch_size/2), len(valid_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    # loader_test = DataLoader( test_dataset, batch_size=min(int(args.batch_size/2),len(test_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t']  )
    # test_y = [ d.by.item() for d in test_dataset ]
    # test_stat = collections.Counter( test_y )

    return loader, loader_val, train_projects, {"train":train_stat, "val":val_stat }

def fetch_dataset(args, dataset_inmemory):
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
            print(p)
            for pf in glob.glob(f"{args.dataset_path}/{p}*"):
               # print(pf)
                n=os.path.basename(pf)
                #print(n)
                projects[n].append(os.path.basename(pf))
                name.append( os.path.basename(pf) )
    return projects, name

import gc
def train_one_test_many(args):
    set_seed(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    _, namelist = projects_dict(args)
    dataset_list = fecth_datalist(args, namelist)
    orginalsavepath = args.saved_model_path
    for k in range(len(namelist)):
        train_on_projects = [ namelist[k] ]
        test_on_projects = [  ]
        for j in range(len(namelist)):
            if j != k:
                test_on_projects.append( namelist[j] )
        args.saved_model_path = f"{orginalsavepath}/{train_on_projects[0]}_fold/"
        train_mode(args, train_on_projects, test_on_projects, dataset_list)
        gc.collect()
        torch.cuda.empty_cache()  
    
    

def train_mode(args, train_project, test_projects, dataset_list):
    os.makedirs( args.saved_model_path, exist_ok=True)
    set_seed(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    logger.info(args.saved_model_path)
    

    loader, loader_val, train_projects, stat = create_dataset(args, train_project, dataset_list)
    json.dump(train_projects, open(os.path.join(args.saved_model_path, "train_projects.json"), "w")  )
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
    logger.info(f"Trainable Parameters Encoder {pytorch_total_params}\n")
    
    encoder.gnn.embedding.fine_tune_embeddings(True)
    if not args.input_model_file == "-1":
            encoder.gnn.embedding.init_embeddings(embeddings)
            logger.info(f"Load Pretraning model {args.input_model_file}")
            encoder.from_pretrained(args.input_model_file + ".pth", device)
  
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model = PredictionLinearModelFineTune(600, num_class, encoder, args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters Model {pytorch_total_params}\n")
    if not args.saved_transfer_model_file == "-1":
        model.load_state_dict( torch.load(args.saved_transfer_model_file, map_location="cpu") )
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

    if args.loss == "CL":
        earlystopping = EarlyStopping(monitor="loss", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    else:
        earlystopping = EarlyStopping(monitor="f1", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    val_res ={}
    for epoch in range(1, args.epochs+1):
        logger.info(" ====epoch === " + str(epoch))
        trainloss = train(args, model, device, loader, optimizer, scheduler )
        logger.info(f"Train Loss {trainloss}")
        evalloss, accuracy_val, precision_val, recall_val, f1_val, result = evalutaion(args, model, device, loader_val, epoch, earlystopping)
        val_res[str(epoch)] = [ trainloss, evalloss, accuracy_val, precision_val, recall_val, f1_val, result ]
    json.dump( val_res, open(os.path.join(args.saved_model_path, "epoch.json"), "w"), indent=6 )
    
    test_res = {}
    logger.info(f"Test Model Build")
    model = build_model(args, device)
    for tp in test_projects:
        logger.info(f"Test Project {tp}")
        dataset_inmemory = dataset_list[tp] 
        test_dataset = dataset_inmemory.data
        loader_test = DataLoader( test_dataset, batch_size=min(int(args.batch_size/2), len(test_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
        testloss, accuracy_test, precision_test, recall_test, f1_test, result_test = test_eval(args, model, device, loader_test)
        test_res[tp] = [ testloss, accuracy_test, precision_test, recall_test, f1_test, result_test ]
    json.dump( test_res, open(os.path.join(args.saved_model_path, "test.json"), "w"), indent=6  )


if __name__ == "__main__":
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
    parser.add_argument('--fixed_size', type=int, default=50,
                        help='train fixed size (default: 50)') 
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="sum",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gat")
    parser.add_argument('--repWay', type=str, default="append", help='seq, append, graph, alpha')
    parser.add_argument('--nonodeembedding', dest='nonodeembedding', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default = 'DV_PDG', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset_path', type=str, default = 'dataset/pittest/', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained_models/context/gat/model_0', help='filename to read the model (if there is any)')
    parser.add_argument('--saved_transfer_model_file', type=str, default= "-1", )
    #parser.add_argument('--pre', type=str, default= "yes", help="warm up training for this task")
    #parser.add_argument('--target_token_path', type=str, default = 'dataset/downstream/java-small/target_word_to_index.json', help='Target Vocab')
    parser.add_argument('--saved_model_path', type = str, default = 'results/mutants_siamese/context', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--sub_token_path', type=str, default = './tokens/jars', help='sub tokens vocab and embedding path')
    parser.add_argument('--emb_file', type=str, default = 'emb_100.txt', help='embedding txt path')
    parser.add_argument('--log_file', type = str, default = 'log.txt', help='log file')
    parser.add_argument('--num_class', type = int, default =2, help='num_class')
    parser.add_argument('--seed', type = int, default =0, help='seed')
    parser.add_argument('--dropratio', type = float, default =0.25, help='drop_ratio')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--warmup_schedule', type=str, default="no",
                        help='warmup')
    parser.add_argument('--task', type=str, default="relevance",
                        help='[killed, relevance]')
    parser.add_argument('--lazy', type=str, default="no",
                        help='save model')
    parser.add_argument("--projects", nargs="+", default=["collections"])
    parser.add_argument("--loss", type=str, default="CE", help='[both, CT, CE, SCL]')

    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

   
    assert len(args.projects) == 1
    logger = get_logger(os.path.join(args.saved_model_path, "log.txt"))
    logger.info('start training!')
    train_one_test_many(args)
    logger.info('finishing training!')

