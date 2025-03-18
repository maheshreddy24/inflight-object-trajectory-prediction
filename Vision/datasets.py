# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# import os
# import torch.nn.functional as F
# from PIL import Image

# device = 'cuda'
# # if torch.cuda.is_available(): device = "cuda"

# class dataset_pre():
#     def __init__(self, path = '/home/mtron_lab/catching/parallel-withoutdepth'):
#         # videos folder 
#         self.ls = os.listdir(path)  # List all file locations
    
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),  # Convert to tensor (H, W, C) -> (C, H, W)
#         ])
#         self.path = path
        
#     def loader(self):
#         train_input = []
#         train_output = []
#         # for each video in the dataset folder 
#         for file in self.ls: #self.ls has the path for all files
#             vid_path = os.path.join(self.path, file)  # Path for each video
#             cap = cv2.VideoCapture(vid_path) 
#             frames = []  # To append frames to this list
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frames.append(frame)  # Append frame to the list

#             # Split frames into input_frames and output_frames (based on your logic, e.g. 1st 10 frames as input, next 10 as output)
#             input_frames = frames[:10]  # Example split, adjust as needed
#             output_frames = frames[10:20]  # Adjust the frame range
#             try:
#                 # Convert frames to tensors
#                 input_tensors = torch.stack([self.transform(frame) for frame in input_frames])  # Stack input frames
#                 output_tensors = torch.stack([self.transform(frame) for frame in output_frames])  # Stack output frames

#                 # Add depth dimension and permute to match expected shape (batch, channels, depth, height, width)
#                 input_tensors = input_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)  # (10, 3, 240, 320) -> (1, 3, 10, 240, 320)
#                 output_tensors = output_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)  # (10, 3, 240, 320) -> (1, 3, 10, 240, 320)

#                 # Transfer tensors to GPU
#                 input_tensors = input_tensors.to(device)
#                 output_tensors = output_tensors.to(device)

#                 # Append tensors to the list (train_input and train_output)
#                 train_input.append(input_tensors) #this will be list of lists
#                 train_output.append(output_tensors)
#             except Exception as error:
#                 continue

#         # After the loop, concatenate the tensors from the list
#         train_input = torch.cat(train_input, dim=0)  # Concatenate along the batch dimension (dim=0)
#         train_output = torch.cat(train_output, dim=0)  # Concatenate along the batch dimension (dim=0)
        
#         return train_input, train_output

# class VideoDataset(Dataset):
#     def __init__(self, input_tensors, output_tensors):
#         self.input_tensors = input_tensors
#         self.output_tensors = output_tensors

#     def __len__(self):
#         return len(self.input_tensors)

#     def __getitem__(self, idx):
#         return self.input_tensors[idx], self.output_tensors[idx]



# if __name__ == '__main__':
#     ds = dataset_pre()
#     train_input, train_output = ds.loader()
#     train_dataset = VideoDataset(train_input, train_output)
#     # test_dataset = VideoDataset(test_input, test_output)

#     # Create DataLoader instances
#     train_dataloader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
#     # test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = True)

#     # Example of how to iterate through the DataLoader
#     for input_batch, output_batch in train_dataloader:
#         print(f"Input batch shape: {input_batch.shape}")
#         print(f"Output batch shape: {output_batch.shape}")
#         break  

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars

device = 'cuda'

class dataset_pre():
    def __init__(self, path='/home/mtron_lab/catching/parallel_with_depth_new'):
        self.ls = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.path = path
        self.train_ls = self.ls[:270]
        self.test_ls = self.ls[270:]

    def test_loader(self):
        test_input = []
        test_output = []
        for file in tqdm(self.test_ls, desc="Loading videos"):
            vid_path = os.path.join(self.path, file)
            cap = cv2.VideoCapture(vid_path) 
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            input_frames = frames[:5]
            output_frames = frames[5:10]
            try:
                input_tensors = torch.stack([self.transform(frame) for frame in input_frames])
                output_tensors = torch.stack([self.transform(frame) for frame in output_frames])

                input_tensors = input_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)
                output_tensors = output_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)

                input_tensors = input_tensors.to(device)
                output_tensors = output_tensors.to(device)

                test_input.append(input_tensors)
                test_output.append(output_tensors)
            except Exception as error:
                continue
        test_input = torch.cat(test_input, dim=0)
        test_output = torch.cat(test_output, dim=0)

        return test_input, test_output


    def train_loader(self):
        train_input = []
        train_output = []
        # Wrap the file iteration with tqdm for progress tracking
        for file in tqdm(self.train_ls, desc="Loading videos"):
            vid_path = os.path.join(self.path, file)
            cap = cv2.VideoCapture(vid_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            input_frames = frames[:5]
            output_frames = frames[5:10]

            try:
                input_tensors = torch.stack([self.transform(frame) for frame in input_frames])
                output_tensors = torch.stack([self.transform(frame) for frame in output_frames])

                input_tensors = input_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)
                output_tensors = output_tensors.unsqueeze(2).permute(2, 1, 0, 3, 4)

                input_tensors = input_tensors.to(device)
                output_tensors = output_tensors.to(device)

                train_input.append(input_tensors)
                train_output.append(output_tensors)

            except Exception as error:
                continue

        train_input = torch.cat(train_input, dim=0)
        train_output = torch.cat(train_output, dim=0)

        return train_input, train_output

class VideoDataset(Dataset):
    def __init__(self, input_tensors, output_tensors):
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


if __name__ == '__main__':
    ds = dataset_pre()
    train_input, train_output = ds.loader()
    train_dataset = VideoDataset(train_input, train_output)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Iterate through DataLoader with tqdm
    for input_batch, output_batch in tqdm(train_dataloader, desc="Iterating over batches"):
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Output batch shape: {output_batch.shape}")
        break
