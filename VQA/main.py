import os
import re
import time
import random
import datetime
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.tensorboard import SummaryWriter

from randaugment import RandomAugment

from tools import *
from vqa_dataset import VQADataset


writer = SummaryWriter(log_dir="runs\VQA_Swin&Bert")  # For tensorboard

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

transform = transforms.Compose([                        
    transforms.Resize([224,224]),
    RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize','ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),  
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])     


train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
test_dataset.update_dict(train_dataset)

print(f"train_dataset: {len(train_dataset)}")
print(f"test_dataset: {len(test_dataset)}")

# Set the batch size
BATCH_SIZE = 128

# Create the train_loader and test_loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create the model
class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.image_encoder = torchvision.models.swin_b(pretrained=True)   # Swin Transformer as the image encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')  # BERT as the text encoder

        self.fc = nn.Sequential(
            nn.Linear(1768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

        # Only train the head of the image encoder
        for name, param in self.image_encoder.named_parameters():
            if "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Only train the pooler of the text encoder
        for name, param in self.text_encoder.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, image, question):
        image_feature = self.image_encoder(image)  # Image features [Batch size, 1000]
        text_encoder_ouputs = self.text_encoder(input_ids=question["input_ids"].squeeze(1), attention_mask = question["attention_mask"].squeeze(1), token_type_ids = question["token_type_ids"].squeeze(1))  # Question features [Batch size, 768]
        
        question_feature = text_encoder_ouputs.pooler_output

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x

model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

# Configurations for the model
config = {
    'init_lr': 0.001,
    'weight_decay': 0.005,
    'n_epochs': 100,
}

# # Define the optimizer and the learning rate scheduler
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=0.)

# Define the number of steps to log the training information
log_every_n_steps = 30
global_step = 0
best_loss = 100000.0

print("-" * 50)
print("Start Training")
start_time = time.time()
for epoch in range(config['n_epochs']):
    losses_train = []
    train_num = 0
    train_true_num = 0
    total_acc = 0
    simple_acc = 0

    # Set the model to train mode
    model.train()
    for i, (image, question, answers, mode_answer) in enumerate(train_loader):
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        
        pred = model(image, question)

        loss = nn.CrossEntropyLoss()(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        writer.add_scalar("Loss_step/train", loss.item(), global_step)
        writer.flush()
        global_step += 1

        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

        # Log the training information
        if i % log_every_n_steps == 0:
            loss_avg = sum(losses_train) / (i+1)

            total_time = time.time() - start_time
            time_str = "Time {},".format(datetime.timedelta(seconds=int(total_time)))
            epoch_str = "Epoch {}/{},".format(epoch+1, config['n_epochs'])
            batch_str = "Batch {}/{},".format(i+1, len(train_loader))
            loss_str = "Train Loss {:.4f},".format(loss_avg)
            acc_str = "VQA acc {:.4f},".format(total_acc / (i+1))
            simple_acc_str = "Simple acc {:.4f}".format(simple_acc / (i+1))
            print(time_str, epoch_str, batch_str, loss_str, acc_str, simple_acc_str)

    # Log the training loss of the epoch
    print('EPOCH: {}, Avrage Train Loss: {:.3f}, VQA Acc: {:.3f}, Simple Acc: {:.3f}'.format(epoch+1, np.mean(losses_train), total_acc / len(train_loader), simple_acc / len(train_loader)))
    writer.add_scalar("Loss_epoch/train", np.mean(losses_train), epoch)
    writer.flush() 
    print("-" * 50)

    # Save the model if the loss is the best
    if np.mean(losses_train) < best_loss:
        best_loss = np.mean(losses_train)
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss
        }
        torch.save(save_obj, os.path.join(".\checkpoints", "vqa_checkpoint_best.pth"))    



# Evaluate the model
# Load the best model
model = VQAModel(n_answer=len(train_dataset.answer2idx))
checkpoint = torch.load(os.path.join(".\checkpoints", "vqa_checkpoint_best.pth"))

model.load_state_dict(checkpoint['model'])
model.to(device)

# Set the model to evaluation mode
model.eval()
submission = []

log_every_n_steps = 100

model.eval()
for i, (image, question) in enumerate(test_loader):
    image, question = image.to(device), question.to(device)
    pred = model(image, question)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

    if i % log_every_n_steps == 0:
        print(f"Processed {i} / {len(test_loader)} samples.")

# Save the submission file
submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
np.save("submission.npy", submission)
print(submission[:10])