import torch
import torch.nn as nn


class MutantPairwiseModel( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, dropratio=0.25, type="supervised"):
        super(MutantPairwiseModel, self).__init__()
        self.encoder = encoder
        self.dense = nn.Linear(in_dim*2, in_dim*2)
        self.dropout = nn.Dropout(dropratio)
        self.out_proj = nn.Linear( in_dim*2, out_dim)
        self.type = type
    
    def forward(self, batch):
        x_s,_,  _ = self.encoder.getVector(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch, batch.ins_length_s)   
        x_t,_,  _ = self.encoder.getVector(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch, batch.ins_length_t)  

        x = torch.cat( (x_s, x_t) , dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        if self.type == "supervised":
            x = torch.relu(x)
            x = self.dropout(x)
            return self.out_proj(x), x
        elif self.type == "feature":
            return x
        else:
            assert False
       
    def loadWholeModel(self, model_file, device, maps={} ):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)




class PredictionLinearSimCLR( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, dropratio=0.25):
        super(PredictionLinearSimCLR, self).__init__()
        self.encoder = encoder
        self.dense = nn.Linear(in_dim , in_dim)
        self.relu =  nn.ReLU()
        self.dropout = nn.Dropout(dropratio)
        self.out_proj = nn.Linear( in_dim, out_dim)
      
    def forward(self, batch):
        x_s,_,  _ = self.encoder.getVector(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch, batch.ins_length_s)   
        x = self.dropout(x_s)
      #  x = self.dense(x)
        x = self.relu(x)
        out = self.out_proj(x_s)
        return x_s, out
    
    def loadWholeModel(self, model_file, device, maps={} ):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)
