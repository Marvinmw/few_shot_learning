from utils.mutantsdataset import PosNegPairDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
dataset = PosNegPairDataset( f"dataset/pittest/collections_25" , "DV_PDG", "collections_25" )
print(len(dataset))
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers = 2)
for d1,d2,l in loader:
    print(len(d1))
    print(len(d1.by))
    print(d2.by)
    print(l)