from pydoc import describe
import sys
# setting path
sys.path.append('../')
from utils.mutantsdataset import MutantKilledDataset, MutantRelevanceDataset, MutantTestRelevanceDataset
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
from utils.classifier import MutantSiameseModel
import collections
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils.ranking import ranking_performance
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
best_f1 = 0
view_test_f1 = 0
criterion = None #nn.CrossEntropyLoss()


def train(args, model, device, loader, optimizer, scheduler):
    global best_f1
    global view_test_f1
    model.train()
    trainloss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()      
        pred = model(batch)    
        if args.num_class == 2:
            y = batch.by
        else:
            y = batch.my

        loss =  criterion( pred, y.float().view(-1, 1)) 
      
        loss.backward()
     #   torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if args.warmup_schedule:
            scheduler.step()  
        trainloss.update(loss.item())
   

    return trainloss.avg

def evalutaion(args, model, device, loader_val, epoch, earlystopping ):
    global best_f1
    global view_test_f1
    model.eval()
    evalloss  = eval(args, model, device, loader_val)
    earlystopping(evalloss, model, performance={"evalloss":evalloss, "epoch":epoch})
      
   # logger.info(f"Best Test {view_test_f1}")
    logger.info(f"Epoch {epoch}, Valid, Eval Loss {evalloss}"  )
    return evalloss

def test_eval(args, device,test_on_projects, test_dataset_dict):
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
    model = MutantSiameseModel(600, args.num_class, encoder, args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters Model {pytorch_total_params}\n")
    f=os.path.join(args.saved_model_path, "saved_model.pt")
    assert os.path.isfile( os.path.join(args.saved_model_path, "saved_model.pt") ), f"Fine tune weights file path is wrong {str(f)}"
    model.load_state_dict( torch.load(os.path.join(args.saved_model_path, "saved_model.pt"), map_location="cpu") )
    model.to(device)
    model.eval()
    sum_res = {}
    for test_p in test_dataset_dict:
        if test_p in test_on_projects:
            try:
                sum_res[test_p] = prediction_similarity( test_dataset_dict[test_p], device,model )
            except Exception as e:
                logger.info( test_p )
                logger.info( e )
    return sum_res

def prediction_similarity(testdataset, device, model):
    testdataset.set_data("bank")
    test_loader_bank = DataLoader(testdataset, batch_size=16, shuffle=False, num_workers = 2,  exclude_keys=["interaction_mid", "node_type", "label_r_mul"])
    bank_mutantid = []
    bank_feature = [ ]
    #creat bank feature
    for batch in test_loader_bank:
        batch = batch.to(device)
        mid_batch = batch.mutantID
        bank_batch = model.forward_once(batch)
        bank_mutantid.append( mid_batch )
        bank_feature.append( bank_batch )
    bank_feature = torch.cat( bank_feature, dim=0 )
    #print(bank_mutantid)
    bank_mutantid = torch.cat( bank_mutantid, dim=0 ) # M X D

    #create query mutant feature
    testdataset.set_data("query")
    test_loader_query = DataLoader(testdataset, batch_size=16, shuffle=False, num_workers = 2, exclude_keys=["interaction_mid", "node_type", "label_r_mul"])
    query_mutantid = [ ]
    query_feature = [ ]
    ground_truth = []
    for batch in test_loader_query:
        batch = batch.to(device)
        mid_batch = batch.mutantID
        query_batch = model.forward_once(batch)
        ground_truth.append( batch.label_r_binary )
        query_mutantid.append( mid_batch )
        query_feature.append( query_batch )
    
  #  print(query_feature[0].shape)
    query_feature = torch.cat( query_feature, dim=0 )
    query_mutantid = torch.cat( query_mutantid, dim=0 ) # N X D
    ground_truth = torch.cat(ground_truth, dim=0)
    N = query_feature.shape[0]
    
  
    scores_list = []
    # query score
    for reference in bank_feature:
        reference = reference.view(1, -1)
        #print(reference.shape)
        repeated = reference.repeat(N, 1)
        #print(repeated.shape)
        similarity = model.score( query_feature, repeated )
        scores_list.append( similarity )
    
    # roc_auc_score
    # print(scores_list[0].shape)
    scorematrix = torch.cat( scores_list, dim=1 )
    # print(scorematrix.shape)
    max_score = torch.mean( scorematrix, 1 ).cpu().detach().numpy()
    ground_truth_np = ground_truth.cpu().detach().numpy().astype(np.float)
    # print(ground_truth_np.shape)
    # print(max_score.shape)
    # print(ground_truth_np)
    res = ranking_performance(ground_truth_np, max_score)
    logger.info(f"{testdataset.project} , {res}")
    return res

        

def eval(args, model, device, loader):
    # y_true = []
    # y_prediction = []
    evalloss = AverageMeter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            if args.num_class == 2:
                y = batch.by
            else:
                y = batch.my
            loss =  criterion( outputs, y.float().view(-1, 1)) 
           
            evalloss.update( loss.item() )         
     
    return evalloss.avg

import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance":
            dataset_inmemory = MutantRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p, probability=0.8 )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list

def fetch_testdata(args, projects):
    dataset_list = {}
    for s, p in enumerate(tqdm(projects, desc="Iteration")):
        if args.task == "killed":
            dataset_inmemory = MutantKilledDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        elif args.task == "relevance":
            dataset_inmemory = MutantTestRelevanceDataset( f"{args.dataset_path}/{p}" , dataname=args.dataset, project=p )
        else:
            assert False, f"wrong task name {args.task}, valid [ killed, relevance ]"
        dataset_list[p] = dataset_inmemory
    return dataset_list


def create_dataset(args, train_projects, dataset_list):
    train_dataset = [ ]
    valid_dataset = [ ]
    test_dataset = []
    data = []
    for tp in train_projects:
            dataset_inmemory = dataset_list[tp] 
            dataset = dataset_inmemory.data
            data.extend( dataset )
    # data=data[:2000] # for local debug
    # args.batch_size=64 # for local debug
    random.shuffle(data)
    data = random.sample(data, int(len(data) * args.data_ratio))
    #data = balanced_subsample(data, y)
    y = [ d.by.item() for d in data ]
    test_size = int(len(data)*0.5)
    val_size = int((len(data)-test_size)*0.3)
    train_size = len(data) - test_size -val_size
    train_dataset = data[:train_size]
    valid_dataset = data[train_size : train_size+ val_size]
    test_dataset=data[  train_size+ val_size: ]
            
    y = [ d.by.item() for d in train_dataset ]
    train_stat = collections.Counter(y)
    weights = [ 1/train_stat[0], 1/train_stat[1]]
    samples_weight = np.array([ weights[d.by.item()] for d in train_dataset])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    #train_dataset = balanced_oversample(train_dataset, y)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])
    val_y = [ d.by.item() for d in valid_dataset ]
    val_stat = collections.Counter( val_y )
    print(len(valid_dataset))
    print(len(test_dataset))
    loader_val = DataLoader( valid_dataset, batch_size=min(int(args.batch_size/2), len(valid_dataset)), shuffle=False, num_workers = args.num_workers,follow_batch=['x_s', 'x_t'])


    return loader, loader_val, train_projects, {"train":train_stat, "val":val_stat}


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



def train_mode(args, train_projects):
    global criterion
    os.makedirs( args.saved_model_path, exist_ok=True)
    set_seed(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    criterion = nn.BCEWithLogitsLoss()  
    orgsavedpat=args.saved_model_path
    dataset_list = fecth_datalist(args, train_projects)

    args.saved_model_path = f"{orgsavedpat}"
    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    logger.info(args.saved_model_path)
    

    loader, loader_val, train_projects, stat = create_dataset(args, train_projects, dataset_list)
    json.dump(train_projects, open(os.path.join(args.saved_model_path, "train_projects.json"), "w")  )
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
    model = MutantSiameseModel(600, num_class, encoder, args.dropratio)
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

    earlystopping = EarlyStopping(monitor="loss", patience=50, verbose=True, path=args.saved_model_path, save_model=save_model)
    
    val_res ={}
    for epoch in range(1, args.epochs+1):
        logger.info(" ====epoch === " + str(epoch))
        trainloss = train(args, model, device, loader, optimizer, scheduler )
        logger.info(f"Train Loss {trainloss}")
        evalloss = evalutaion(args, model, device, loader_val, epoch, earlystopping)
        val_res[str(epoch)] = [ trainloss, evalloss]
    json.dump( val_res, open(os.path.join(args.saved_model_path, "epoch.json"), "w") )    
   

import gc
def train_one_test_many(args):
    global criterion
    global contrastive_loss
    global self_contrastive_loss
    global suploss
    set_seed(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    _, namelist = projects_dict(args)
    test_dataset_dict = fetch_testdata(args, namelist )
    orginalsavepath = args.saved_model_path
    if args.fine_tune == "yes":
        for k in range(len(namelist)):
            train_on_projects = [ namelist[k] ]
            test_on_projects = [  ]
            for j in range(len(namelist)):
                if j != k:
                    test_on_projects.append( namelist[j] )
            args.saved_model_path = f"{orginalsavepath}/{train_on_projects[0]}_fold/"
            if args.check_failed == "yes":
                if os.path.isfile( os.path.join(args.saved_model_path, "saved_model.pt" ) ):
                    continue
            try:
                logger.info(f"fine tune {train_on_projects}")
                train_mode(args, train_on_projects )
                # run test
                logger.info("prediction")
                sum_res = test_eval(args, device, test_on_projects, test_dataset_dict)
                json.dump( sum_res, open(os.path.join(args.saved_model_path, "few_shot_test.json"), "w") , indent=6) 
            except Exception as e:
                logger.info(e)     
            gc.collect()
            torch.cuda.empty_cache()  
    else:
        logger.info("prediction")
        args.saved_model_path =  os.path.dirname( args.saved_transfer_model_file )
        sum_res = test_eval(args, device, namelist, test_dataset_dict)
        json.dump( sum_res, open(os.path.join(args.saved_model_path, "few_shot_test.json"), "w"), indent=6 ) 
    

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
    parser.add_argument('--task', type=str, default="killed",
                        help='[killed, relevance]')
    parser.add_argument('--lazy', type=str, default="no",
                        help='save model')
    parser.add_argument("--projects", nargs="+", default=["collections"])

   # parser.add_argument("--loss", type=str, default="CE", help='[both, CL, CE, SCL]')
    parser.add_argument("--data_ratio", type=float, default=1.0, help='used dataset set size')
    parser.add_argument("--fine_tune",  type=str, default="no",
                        help='[yes, no]')
    parser.add_argument("--check_failed",  type=str, default="yes",
                        help='[yes, no]')
    
    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args = parser.parse_args( )
    with open(args.saved_model_path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

   
    assert len(args.projects) == 1
    logger = get_logger(os.path.join(args.saved_model_path, "log.txt"))

    train_one_test_many(args)


