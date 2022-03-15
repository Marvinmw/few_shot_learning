import torch
import torch.nn as nn


class MutantPairwiseModel( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, dropratio=0.25, type="supervised"):
        super(MutantPairwiseModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropratio)
        self.out_proj = nn.Linear( in_dim, out_dim)
        self.type = type
    
    def forward(self, batch):
        x_s,_,  _ = self.encoder.getVector(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch, batch.ins_length_s)   
        x_t,_,  _ = self.encoder.getVector(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch, batch.ins_length_t)  
        feature = x_s+ x_t 
        x = feature
        if self.type == "supervised":
            x = torch.relu(x)
            x = self.dropout(x)
            return self.out_proj(x), feature
        elif self.type == "feature":
            return x
        else:
            assert False
       
    def loadWholeModel(self, model_file, device, maps={} ):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)





class MutantSiameseModel( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, dropratio=0.25, type="combination"):
        super(MutantSiameseModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropratio)
        self.out_proj = nn.Linear( in_dim*4, 1 )
      #  self.classifier = torch.sigmoid()
        self.type = type
    
    def forward(self, batch):
        x_s,_,  _ = self.encoder.getVector(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch, batch.ins_length_s)   
        x_t,_,  _ = self.encoder.getVector(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch, batch.ins_length_t)  
        if self.type == "combination":
            x = torch.cat( ( x_s, torch.abs(x_s-x_t), torch.square(x_s-x_t), x_t ), dim=1 )
        else:
            x =  torch.abs(x_s-x_t)
        x = self.out_proj(x)
        return x

    def forward_once(self, batch):
        x,_,  _ = self.encoder.getVector(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length)
        return x
    
    def score(self, f1, f2):
        if self.type == "combination":
            x = torch.cat( ( f1, torch.abs(1-f2), torch.square(f1-f2), f2 ), dim=1 )
        else:
            x =  torch.abs(f1 - f2)
        score = torch.sigmoid(self.out_proj(x))
        return score
        

       
    def loadWholeModel(self, model_file, device, maps={} ):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)
