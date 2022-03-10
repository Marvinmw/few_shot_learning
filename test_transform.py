from utils.mutantsdataset import MyOwnDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
dataset = MyOwnDataset( f"dataset/pittest/collections_25" , "DV_PDG", "collections_25", transform=T.RandomFlip(1) )
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers = 2)
for d1,d2 in loader:
    print(d1)
    print(d2)