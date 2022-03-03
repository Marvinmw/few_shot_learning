import torch
import torch.nn as nn

# class PredictionLinearModel( nn.Module ):
#     def __init__(self,in_dim, out_dim):
#         super(PredictionLinearModel, self).__init__()
#         self.linear1 = nn.Linear(in_dim, out_dim)
      
#         #self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)
   
#     def forward(self, x):
#         # if len(argv) == 1:
#         #     x = argv[0]
#         # if len(argv) == 2:
#         #     x1, x2 = argv[0], argv[1]
#         #     x = torch.sigmoid( self.alpha ) * x1  + ( 1 - torch.sigmoid(self.alpha) ) * x2
#         return self.linear1(x)

#     # def getVector(self, *argv):
#     #     if len(argv) == 1:
#     #         x = argv[0]
#     #     if len(argv) == 2:
#     #         x1, x2 = argv[0], argv[1]
#     #         x = torch.sigmoid( self.alpha ) * x1  + ( 1 - torch.sigmoid(self.alpha) ) * x2
#     #     return x

class PredictionLinearModelFineTune( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, addedMutantType=False, dropratio=0.25):
        super(PredictionLinearModelFineTune, self).__init__()
        self.encoder = encoder
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = 1 - self.alpha
        self.addedMutantType = addedMutantType
        if addedMutantType:
            self.dense = nn.Linear(in_dim*2*2+in_dim//3*3, in_dim*2)
        else:
            self.dense = nn.Linear(in_dim*2*2, in_dim*2)
        self.dropout = nn.Dropout(dropratio)
        #self.out_proj = nn.Linear( in_dim*2, out_dim)
        self.out_proj1 = nn.Linear( in_dim, out_dim)
        self.out_proj2 = nn.Linear( in_dim*2, out_dim)
        self.type_embeddings = nn.Embedding(9, in_dim//3, padding_idx=0)
        self.oprand1_embeddings =  nn.Embedding(34, in_dim//3, padding_idx=0)
        self.oprand2_embeddings =  nn.Embedding(34, in_dim//3, padding_idx=0)
       # self.bilinear = nn.Bilinear(600, 600, 600, bias=False)
        #self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)
   
    def forward(self, batch, args=None):
        x_s,_,  _ = self.encoder.getVector(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch, batch.ins_length_s)   
        x_t,_,  _ = self.encoder.getVector(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch, batch.ins_length_t)  
        
        #x0 = torch.square(torch.sub(x_s, x_t))
        if args.loss == "both":
            x2 = torch.abs( torch.sub(x_s, x_t) )
            return self.out_proj1(x2), x_s, x_t
        else:
            x2 = torch.abs( torch.cat((x_s, x_t)) )
            return self.out_proj2(x2), x_s, x_t
        # if self.addedMutantType:
        #     mx=self.type_embeddings(batch.type) 
        #     op1=self.oprand1_embeddings(batch.operand1)
        #     op2=self.oprand2_embeddings(batch.operand2)
        #     x = torch.cat( (x_s, x_t, x2, x0, mx, op1, op2) , dim=1)
        # else:
        #     x = torch.cat( (x_s, x_t,x2, x0) , dim=1)
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        
    
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
