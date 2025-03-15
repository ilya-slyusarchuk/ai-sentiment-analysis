import os
import argparse
import torch
from meld_dataset import prepare_dataloader
from models import MultiModalFusion, MultiModalTrainer
from tqdm import tqdm
import json

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "../dataset/train")
SM_CHANNEL_VALIDATION = os.environ.get("SM_CHANNEL_VALIDATION", "../dataset/dev")
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "../dataset/test")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  
  # Data directories
  parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
  parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
  parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
  parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
  
  return parser.parse_args()
  
def main():
  args = parse_args()
  
  # Add support for Apple Silicon MPS
  if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS acceleration")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU acceleration")
  else:
    device = torch.device("cpu")
    print("Using CPU for training")
  
  # Tracking memory if GPU is available
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Memory used: {memory_used:.2f} GB")
    
  train_loader, dev_loader, test_loader = prepare_dataloader(train_csv_path=os.path.join(args.train_dir, 'train_sent_emo.csv'), train_video_dir=os.path.join(args.train_dir, 'train_splits/'), dev_csv_path=os.path.join(args.val_dir, 'dev_sent_emo.csv'), dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete/'), test_csv_path=os.path.join(args.test_dir, 'test_sent_emo.csv'), test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_complete/'))
  
  model = MultiModalFusion().to(device)
  trainer = MultiModalTrainer(model, train_loader, dev_loader)
  
  best_validation_loss = float('inf')
  
  metrics_data = {
    "train_losses": [],
    "validation_losses": [],
    "test_losses": [],
    "test_metrics": [],
    "epochs": [],
  }
  
  for epoch in tqdm(range(args.epochs), desc="Epochs"):
    train_losses = trainer.train_epoch()
    dev_losses, dev_metrics = trainer.validate(dev_loader)
    
    metrics_data["train_losses"].append(train_losses['total'])
    metrics_data["validation_losses"].append(dev_losses['total'])
    metrics_data["epochs"].append(epoch)
    
    print(json.dumps({
      "metrics": [
        {"Name": "train:loss", "Value": train_losses['total']},
        {"Name": "validation:loss", "Value": dev_losses['total']},
        {"Name": "validation:emotion:precision", "Value": dev_metrics['emotion_precision']},
        {"Name": "validation:sentiment:precision", "Value": dev_metrics['sentiment_precision']},
        {"Name": "validation:emotion:accuracy", "Value": dev_metrics['emotion_accuracy']},
        {"Name": "validation:sentiment:accuracy", "Value": dev_metrics['sentiment_accuracy']}
        ],
      }))
    
    if torch.cuda.is_available():
      memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
      print(f"Peak memory used: {memory_used:.2f} GB")
    
    if dev_losses['total'] < best_validation_loss:
      best_validation_loss = dev_losses['total']
      torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
      
  # Evaluate on test set
  test_losses, test_metrics = trainer.validate(test_loader)
  metrics_data["test_losses"].append(test_losses['total'])
  metrics_data["test_metrics"].append(test_metrics)
  
  print(json.dumps({
    "metrics": [
        {"Name": "test:loss", "Value": test_losses['total']},
        {"Name": "test:emotion:precision", "Value": test_metrics['emotion_precision']},
        {"Name": "test:sentiment:precision", "Value": test_metrics['sentiment_precision']},
        {"Name": "test:emotion:accuracy", "Value": test_metrics['emotion_accuracy']},
        {"Name": "test:sentiment:accuracy", "Value": test_metrics['sentiment_accuracy']}
      ],
  }))
      
if __name__ == "__main__":
  main()