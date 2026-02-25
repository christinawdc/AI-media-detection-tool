import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from multiprocessing import freeze_support
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler 

# ======================================================
# CONFIGURATION - TUNED FOR 128 BATCH SIZE
# ======================================================
# Force PyTorch to manage memory fragmentation better
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATASET_DIR = "./dataset"
REAL_DIR = os.path.join(DATASET_DIR, "Real")
AI_DIR   = os.path.join(DATASET_DIR, "AI")

IMAGE_SIZE = 224
BATCH_SIZE = 128         # Physical batch size
ACCUMULATION_STEPS = 2    # Effective batch size = 256
EPOCHS = 10
LEARNING_RATE = 1e-3 

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True 

# ======================================================
# DATASET & LOGGING
# ======================================================
def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open("model_output/training_summary.txt", "a") as f:
        f.write(formatted_msg + "\n")

class AIVsRealDataset(Dataset):
    def __init__(self, real_dir, ai_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Recursively find images in Real directory and subdirectories
        real_images = []
        for root, dirs, files in os.walk(real_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    real_images.append(os.path.join(root, f))
        
        # Recursively find images in AI directory and subdirectories
        ai_images = []
        for root, dirs, files in os.walk(ai_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    ai_images.append(os.path.join(root, f))
        
        # Use ALL images - no ratio restriction
        for path in real_images: self.samples.append((path, 0))
        for path in ai_images: self.samples.append((path, 1))
        random.shuffle(self.samples)
        
        log_to_file(f"Dataset loaded: {len(real_images)} Real, {len(ai_images)} AI (Total: {len(self.samples)})")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, label
        except: 
            return self.__getitem__(random.randint(0, len(self.samples)-1))

# ======================================================
# TRAINING ENGINE
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for i, (images, labels) in enumerate(pbar):
        # non_blocking=True helps move data while CPU is busy
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with autocast(device_type='cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
            # Scaling loss for accumulation
            loss = loss / ACCUMULATION_STEPS 

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUMULATION_STEPS
        
        # Monitor VRAM in the progress bar
        if i % 10 == 0:
            vram = torch.cuda.memory_reserved() / 1E9
            pbar.set_postfix(loss=f"{loss.item()*ACCUMULATION_STEPS:.4f}", vram=f"{vram:.1f}GB")
        
    return total_loss / len(loader)

if __name__ == "__main__":
    freeze_support()
    os.makedirs("model_output", exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AIVsRealDataset(REAL_DIR, AI_DIR, transform=train_transform)
    
    # 24GB RAM allows us to be aggressive with preloading
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, # Balanced for Ryzen 7
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )

    log_to_file(f"Starting Session | Batch: {BATCH_SIZE} | Effective: {BATCH_SIZE*ACCUMULATION_STEPS}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        scheduler.step()
        log_to_file(f"Epoch {epoch+1} Complete | Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"model_output/resnet50_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "model_output/resnet50_final.pth")
    log_to_file("Training Complete.")