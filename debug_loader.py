import torch
from torch.utils.data import DataLoader
from kinexar.TRAINING.minimal_dataset import MinimalShardedDataset, minimal_collate
import time

def debug_loader():
    DATA_DIR = "/home/aryan/work/startup/KINEXAR/src/kinexar/DATA/train"
    print("Initializing dataset...")
    dataset = MinimalShardedDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, # Use 0 for debugging to avoid multiprocessing issues
        collate_fn=minimal_collate
    )
    
    print("Fetching first batch...")
    t0 = time.time()
    for i, batch in enumerate(loader):
        print(f"Batch {i} fetched in {time.time() - t0:.4f}s")
        print(f"Keys: {batch.keys()}")
        print(f"Feat shape: {batch['feat'].shape}")
        break

if __name__ == "__main__":
    debug_loader()
