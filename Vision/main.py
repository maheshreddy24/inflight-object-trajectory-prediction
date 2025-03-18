from datasets import dataset_pre, VideoDataset
from models import Network
from optimisation import Optimisation
from torch.utils.data import DataLoader

def main():
    fs = dataset_pre()
    train_input, train_output = fs.train_loader()
    test_input, test_output = fs.test_loader()
    

    train_dataset = VideoDataset(train_input, train_output)
    test_dataset = VideoDataset(test_input, test_output)

    # Create DataLoader instances
    train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = True)

    # Example of how to iterate through the DataLoader
    for input_batch, output_batch in train_dataloader:
        print(f"Train Input batch shape: {input_batch.shape}")
        print(f"Train Output batch shape: {output_batch.shape}")
        break  
    for input_batch, output_batch in test_dataloader:
        print(f"Test Input batch shape: {input_batch.shape}")
        print(f"Test Output batch shape: {output_batch.shape}")
        break
    model = Network()
    # def __init__(self, model, device, epochs, learning_rate, train_dataloader):
    opt = Optimisation(model, 'cuda', 20, 0.002, train_dataloader, test_dataloader)
    opt.train()
    # opt.evaluation()
    
    import torch
    torch.save(test_input, 'test_input.pth')
    torch.save(test_output, 'test_output.pth')
    
if __name__ == "__main__":
    main()