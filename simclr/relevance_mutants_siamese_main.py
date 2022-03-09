from matplotlib import collections
#from utils.mutantsdataset_siamese import MutantsDataset
from utils.mutantsdataset_relevance import MutantsDatasetSingleData, balanced_oversample
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
from utils.probing_classifier import PredictionLinearSimCLR
import collections
import random
import gc
# try:
#     from transformers import get_linear_schedule_with_warmup as linear_schedule
# except:
#     print("import WarmupLinearSchedule instead of get_linear_schedule_with_warmup")
#     from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
#     def linear_schedule(optimizer, num_warmup_steps=100, num_training_steps=100):
#         return get_linear_schedule_with_warmup(optimizer, warmup_steps=num_warmup_steps,
#                                                     t_total=num_training_steps)
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



def train(args, model, device, loader, train_optimizer ):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)  
        features1,  out_1  = model(batch)    
        features2,  out_2  = model(batch)    
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) /  args.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 *  args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) /  args.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(args,net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, args.c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, args.c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
import glob
def fecth_datalist(args, projects):
    dataset_list = {}
    for p in projects:
        #set up dataset dataset_path
        dataset_inmemory = MutantsDatasetSingleData( args.dataset_path , dataname=args.dataset, project=p)
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
    os.makedirs( args.saved_model_path, exist_ok=True)
    if args.graph_pooling == "set2set":
        args.graph_pooling = [2, args.graph_pooling]

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    
    projects, namelist = projects_dict(args)
    
    train_projects = []
    klist = list( projects.keys() )
    for pp in klist:
                train_projects.extend(projects[pp])
    orgsavedpat=args.saved_model_path

    dataset_list = fecth_datalist(args, namelist)

    args.saved_model_path = f"{orgsavedpat}"
    if not os.path.isdir(args.saved_model_path):
            os.makedirs(args.saved_model_path)
    print(args.saved_model_path)
    

    loader, loader_val,  loader_test, train_projects, stat = create_dataset(args, train_projects, dataset_list)
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
    print(f"Trainable Parameters Encoder {pytorch_total_params}\n")
    
    encoder.gnn.embedding.fine_tune_embeddings(True)
    if not args.input_model_file == "-1":
            encoder.gnn.embedding.init_embeddings(embeddings)
            print(f"Load Pretraning model {args.input_model_file}")
            encoder.from_pretrained(args.input_model_file + ".pth", device)
  
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal Number of Parameters of Model, {pytorch_total_params}")
    model = PredictionLinearSimCLR(600, 128,encoder,False if args.mutant_type == "no" else True,args.dropratio)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters Model {pytorch_total_params}\n")
    # if not args.saved_model_file == "-1":
    #     model.load_state_dict( torch.load(args.saved_model_file, map_location="cpu"), strict=False )
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.decay ) 
    for epoch in range(1, args.epochs+1):
        print(" ====epoch === " + str(epoch))
        train_loss  = train(args, model, device, loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
       

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

