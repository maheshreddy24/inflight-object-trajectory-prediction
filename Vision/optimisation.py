# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# import os
# from datetime import datetime

# class Optimisation():
#     def __init__(self, model, device, epochs, learning_rate, train_dataloader, test_dataloader):
#         self.model = model.to(device)
#         self.device = device
#         self.lr = learning_rate
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
#         self.epochs = epochs
#         self.loss_list = []
#         self.train_dataloader = train_dataloader
#         self.test_data_loader = test_dataloader

#         # Create experiments directory with current date
#         current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#         self.checkpoint_dir = os.path.join('experiments', current_date)
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def save_checkpoint(self, epoch):
#         """Save model checkpoint."""
#         checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss_list': self.loss_list,
#         }, checkpoint_path)
#         print(f"Checkpoint saved at: {checkpoint_path}")

#     def evaluation(self):
#         self.model.eval()  # Set the model to evaluation mode
#         total_loss = 0  # Accumulator for the loss
        
#         with torch.no_grad():  # No need to calculate gradients during evaluation
#             with tqdm(self.test_data_loader, unit="batch") as tepoch:
#                 for input_batch, output_batch in tepoch:
#                     input_batch = input_batch.to(self.device)
#                     output_batch = output_batch.to(self.device)

#                     # Forward pass
#                     output = self.model(input_batch)

#                     # Calculate loss
#                     loss = self.criterion(output, output_batch)
#                     total_loss += loss.item()

#                     # Update tqdm with current batch loss
#                     tepoch.set_postfix(batch_loss=loss.item())

#         # Calculate average loss
#         avg_loss = total_loss / len(self.test_data_loader)
#         print('Evaluation Loss:', avg_loss)

        
#     def train(self):
#         for epoch in range(1, self.epochs + 1):
#             self.model.train()  # Set the model to training mode
#             train_loss = 0
#             batch_losses = []
#             print(f'Epoch {epoch}/{self.epochs}')

#             # Use tqdm for progress visualization
#             with tqdm(self.train_dataloader, unit="batch") as tepoch:
#                 for input_batch, output_batch in tepoch:
#                     tepoch.set_description(f"Epoch {epoch}")

#                     # Move batches to the correct device
#                     input_batch = input_batch.to(self.device)
#                     output_batch = output_batch.to(self.device)

#                     # Forward pass
#                     output = self.model(input_batch)

#                     # # Ensure correct shape for comparison
#                     # output = output[:, :, :5, :, :]
#                     # output_batch = output_batch[:, :, :5, :, :]

#                     # Shape validation
#                     if output.shape != output_batch.shape:
#                         raise ValueError(f"Output shape {output.shape} does not match target shape {output_batch.shape}")

#                     # Compute loss and backpropagate
#                     loss = self.criterion(output, output_batch)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()

#                     # Update progress bar and batch loss
#                     batch_losses.append(loss.item())
#                     tepoch.set_postfix(batch_loss=loss.item())

#             # Calculate average loss for the epoch
#             epoch_loss = sum(batch_losses) / len(batch_losses)
#             self.loss_list.append(epoch_loss)
#             print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

#             # Save checkpoint every 10th epoch
#             if epoch % 10 == 0:
#                 self.save_checkpoint(epoch)
#                 # self.evaluation()
#         # Save final checkpoint
#         self.save_checkpoint(self.epochs)
#         return output

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime

# class Optimisation():
#     def __init__(self, model, device, epochs, learning_rate, train_dataloader, test_dataloader):
#         self.model = model.to(device)
#         self.device = device
#         self.lr = learning_rate
#         self.epochs = epochs
#         self.loss_list = []
#         self.train_dataloader = train_dataloader
#         self.test_dataloader = test_dataloader

#         self.criterion = nn.MSELoss()

#         # Optimizer with Momentum and Weight Decay
#         self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

#         # Learning Rate Scheduler (reduces LR when loss plateaus)
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

#         # Create experiments directory with current date
#         current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#         self.checkpoint_dir = os.path.join('experiments', current_date)
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def save_checkpoint(self, epoch):
#         """Save model checkpoint."""
#         checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss_list': self.loss_list,
#         }, checkpoint_path)
#         print(f"Checkpoint saved at: {checkpoint_path}")

#     def evaluation(self):
#         self.model.eval()
#         total_loss = 0
        
#         with torch.no_grad():
#             with tqdm(self.test_dataloader, unit="batch") as tepoch:
#                 for input_batch, output_batch in tepoch:
#                     input_batch = input_batch.to(self.device)
#                     output_batch = output_batch.to(self.device)

#                     # Forward pass
#                     output = self.model(input_batch)

#                     # Calculate loss
#                     loss = self.criterion(output, output_batch)
#                     total_loss += loss.item()

#                     # Update tqdm with current batch loss
#                     tepoch.set_postfix(batch_loss=loss.item())

#         # Calculate average loss
#         avg_loss = total_loss / len(self.test_dataloader)
#         print('Evaluation Loss:', avg_loss)

#         # Step the scheduler based on evaluation loss
#         self.scheduler.step(avg_loss)

#     def train(self):
#         for epoch in range(1, self.epochs + 1):
#             self.model.train()
#             batch_losses = []
#             print(f'\nEpoch {epoch}/{self.epochs}')

#             with tqdm(self.train_dataloader, unit="batch") as tepoch:
#                 for input_batch, output_batch in tepoch:
#                     tepoch.set_description(f"Epoch {epoch}")

#                     # Move batches to the correct device
#                     input_batch = input_batch.to(self.device)
#                     output_batch = output_batch.to(self.device)

#                     # Forward pass
#                     output = self.model(input_batch)

#                     # Shape validation
#                     if output.shape != output_batch.shape:
#                         raise ValueError(f"Output shape {output.shape} does not match target shape {output_batch.shape}")

#                     # Compute loss
#                     loss = self.criterion(output, output_batch)
#                     self.optimizer.zero_grad()
#                     loss.backward()

#                     # Gradient Clipping
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

#                     self.optimizer.step()

#                     batch_losses.append(loss.item())
#                     tepoch.set_postfix(batch_loss=loss.item())

#             # Calculate average loss for the epoch
#             epoch_loss = sum(batch_losses) / len(batch_losses)
#             self.loss_list.append(epoch_loss)
#             print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

#             # Save checkpoint and evaluate every 10 epochs
#             if epoch % 10 == 0:
#                 self.save_checkpoint(epoch)
#                 self.evaluation()

#         # Save final checkpoint
#         self.save_checkpoint(self.epochs)
#         return output



import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from datetime import datetime

class Optimisation():
    def __init__(self, model, device, epochs, learning_rate, train_dataloader, test_dataloader):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_list = []
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = nn.MSELoss()

        # Using AdamW Optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        # Gradient Scaler for mixed precision training
        self.scaler = GradScaler()

        # Create experiments directory with current date
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.checkpoint_dir = os.path.join('experiments', current_date)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list,
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def evaluation(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(self.test_dataloader, unit="batch") as tepoch:
                for input_batch, output_batch in tepoch:
                    input_batch = input_batch.to(self.device)
                    output_batch = output_batch.to(self.device)

                    # Use mixed precision for evaluation
                    with autocast():
                        output = self.model(input_batch)
                        loss = self.criterion(output, output_batch)
                        total_loss += loss.item()

                    tepoch.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(self.test_dataloader)
        print('Evaluation Loss:', avg_loss)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            batch_losses = []
            print(f'\nEpoch {epoch}/{self.epochs}')

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for input_batch, output_batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    input_batch = input_batch.to(self.device)
                    output_batch = output_batch.to(self.device)

                    # Mixed Precision Training
                    with autocast():
                        output = self.model(input_batch)

                        if output.shape != output_batch.shape:
                            raise ValueError(f"Output shape {output.shape} does not match target shape {output_batch.shape}")

                        loss = self.criterion(output, output_batch)

                    # Backpropagation
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()

                    # Gradient Clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    batch_losses.append(loss.item())
                    tepoch.set_postfix(batch_loss=loss.item())

            # Average loss calculation
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.loss_list.append(epoch_loss)
            print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

            # Scheduler step
            self.scheduler.step()

            # Save and evaluate every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                self.evaluation()

        self.save_checkpoint(self.epochs)
        return output
