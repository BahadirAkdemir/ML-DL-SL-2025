import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm

from pytorch.FaceDataset import AgePredictionDataset
from pytorch.model import Model
from pytorch.lars_optimizer import LARSOptimizer
from config_args import args

class StudentModel(nn.Module):
    """A smaller student model for distillation."""
    def __init__(self, num_classes=1000):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def add_kd_loss(student_logits, teacher_logits, temperature=1.0):
    """Compute knowledge distillation loss."""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    return kd_loss

def train_distillation():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create teacher model (pretrained SimCLRv2)
    teacher_model = Model(num_classes=args.num_classes)
    #teacher_model.load_state_dict(torch.load(args.teacher_checkpoint))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set teacher to eval mode
    
    # Create student model
    student_model = StudentModel(num_classes=args.num_classes)
    student_model = student_model.to(device)
    
    # Get dataset and dataloader
    #train_dataset = get_dataset(args, is_train=True)
    transform = transforms.Compose([
        transforms.Resize((args.training_image_size, args.training_image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = AgePredictionDataset(csv_path=args.train_csv_file, root_dir=args.train_root_folder,
                                         transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    # Setup optimizer
    if args.optimizer == 'lars':
        optimizer = LARSOptimizer(
            student_model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=['batch_normalization', 'bias']
        )
    else:
        optimizer = optim.SGD(
            student_model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.train_epochs,
        eta_min=0
    )
    
    # Training loop
    for epoch in range(args.train_epochs):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train_epochs}')
        for images, _ in pbar:
            images = images.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                teacher_logits = teacher_outputs[1]  # Get supervised head outputs
            
            # Get student predictions
            student_logits = student_model(images)
            
            # Compute distillation loss
            loss = add_kd_loss(student_logits, teacher_logits, temperature=args.temperature)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.model_dir, f'student_model_epoch_{epoch+1}.pt'))

def main():
    # Train with distillation
    train_distillation()

if __name__ == '__main__':
    main() 