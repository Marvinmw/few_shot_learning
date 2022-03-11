from utils.mutantsdataset import MyOwnDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
dataset = MyOwnDataset( f"dataset/pittest/collections_25" , "DV_PDG", "collections_25" )
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers = 2)
for d1,d2,l in loader:
    print(d1.by)
    print(d2.by)
    print(l)