import os
import argparse
import torch
from meld_dataset import prepare_dataloader
from models import MultiModalFusion, MultiModalTrainer
from tqdm import tqdm
import json
from sklearn.metrics import precision_score, accuracy_score

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
    
  train_loader, dev_loader, test_loader = prepare_dataloader(train_csv_path=os.path.join(args.train_dir, 'train_sent_emo.csv'), train_video_dir=os.path.join(args.train_dir, 'train_splits/'), dev_csv_path=os.path.join(args.val_dir, 'dev_sent_emo.csv'), dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete/'), test_csv_path=os.path.join(args.test_dir, 'test_sent_emo.csv'), test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_complete/'), batch_size=args.batch_size)
  
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
  
  # Modify trainer's train_epoch method to use tqdm
  def train_epoch_with_progress(trainer):
    trainer.model.train()
    
    running_loss = {
      'total': 0.0,
      'emotions': 0.0,
      'sentiments': 0.0
    }
    
    # Get device
    device = next(trainer.model.parameters()).device
    device_type = device.type
    
    # Force model to use float32 for MPS compatibility
    if device_type == 'mps':
      for param in trainer.model.parameters():
        if param.data.dtype == torch.float64:
          param.data = param.data.float()  # Convert to float32
    
    # Use mixed precision for speedup - updated syntax
    # Only use scaler for CUDA, not for MPS due to float64 conversion issues
    scaler = torch.amp.GradScaler() if device_type == 'cuda' else None
    
    # Enable mixed precision only for CUDA, not MPS
    use_amp = device_type == 'cuda'
    
    total_samples = 0
    
    # Add progress bar for batches
    progress_bar = tqdm(trainer.train_loader, desc="Training batches", leave=False)
    
    for batch in progress_bar:
      batch_size = batch['text_input']['input_ids'].size(0)
      total_samples += batch_size
      
      text_inputs = {
        'input_ids': batch['text_input']['input_ids'].to(device),
        'attention_mask': batch['text_input']['attention_mask'].to(device)
      }
      video_frames = batch['video_frames'].to(device)
      audio_data = batch['audio_features'].to(device)
      emotion_labels = batch['emotion_labels'].to(device)
      sentiment_labels = batch['sentiment_labels'].to(device)
      
      trainer.optimizer.zero_grad()
      
      # Use automatic mixed precision where supported
      if use_amp:
        with torch.autocast(device_type=device_type):
          outputs = trainer.model(text_inputs, video_frames, audio_data)
          emotion_loss = trainer.emotion_criterion(outputs['emotions'], emotion_labels)
          sentiment_loss = trainer.sentiment_criterion(outputs['sentiments'], sentiment_labels)
          loss = emotion_loss + sentiment_loss
          
        if scaler:
          scaler.scale(loss).backward()
          torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
          scaler.step(trainer.optimizer)
          scaler.update()
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
          trainer.optimizer.step()
      else:
        outputs = trainer.model(text_inputs, video_frames, audio_data)
        emotion_loss = trainer.emotion_criterion(outputs['emotions'], emotion_labels)
        sentiment_loss = trainer.sentiment_criterion(outputs['sentiments'], sentiment_labels)
        loss = emotion_loss + sentiment_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        trainer.optimizer.step()
      
      # Update running losses
      running_loss['total'] += loss.item() * batch_size
      running_loss['emotions'] += emotion_loss.item() * batch_size
      running_loss['sentiments'] += sentiment_loss.item() * batch_size
      
      # Update progress bar with current loss
      progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate average losses
    for key in running_loss:
      running_loss[key] /= total_samples
      
    return running_loss

  # Add progress bar for validation
  def validate_with_progress(trainer, data_loader, phase="validation"):
    trainer.model.eval()
    
    losses = {
      'total': 0.0,
      'emotions': 0.0,
      'sentiments': 0.0
    }
    
    # Get device 
    device = next(trainer.model.parameters()).device
    
    # Ensure float32 for MPS compatibility
    if device.type == 'mps':
      for param in trainer.model.parameters():
        if param.data.dtype == torch.float64:
          param.data = param.data.float()
    
    all_emotion_predictions = []
    all_sentiment_predictions = []
    all_emotion_labels = []
    all_sentiment_labels = []
    
    # Add progress bar
    progress_bar = tqdm(data_loader, desc=f"{phase.capitalize()} batches", leave=False)
    
    with torch.inference_mode():
      for batch in progress_bar:
        text_inputs = {
          'input_ids': batch['text_input']['input_ids'].to(device),
          'attention_mask': batch['text_input']['attention_mask'].to(device)
        }
        video_frames = batch['video_frames'].to(device)
        audio_data = batch['audio_features'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        sentiment_labels = batch['sentiment_labels'].to(device)
        
        outputs = trainer.model(text_inputs, video_frames, audio_data)
      
        emotion_loss = trainer.emotion_criterion(outputs['emotions'], emotion_labels)
        sentiment_loss = trainer.sentiment_criterion(outputs['sentiments'], sentiment_labels)
        
        loss = emotion_loss + sentiment_loss
        
        losses['total'] += loss.item()
        losses['emotions'] += emotion_loss.item()
        losses['sentiments'] += sentiment_loss.item()
        
        all_emotion_predictions.extend(outputs['emotions'].argmax(dim=1).cpu().numpy())
        all_sentiment_predictions.extend(outputs['sentiments'].argmax(dim=1).cpu().numpy())
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    average_loss = {
      k: v / len(data_loader) for k, v in losses.items()
    }
    
    # Precision is the ratio of true positives to all positive predictions
    # Weighted precision takes into account the class imbalance
    emotion_precision = precision_score(all_emotion_labels, all_emotion_predictions, average='weighted')
    sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_predictions, average='weighted')
    # Accuracy is the ratio of correct predictions to all predictions
    emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_predictions)
    sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_predictions)
    
    if hasattr(trainer, 'log_metrics'):
      trainer.log_metrics(average_loss, {
        'emotion_precision': emotion_precision,
        'sentiment_precision': sentiment_precision,
      }, phase)
    
    if phase == "validation" and hasattr(trainer, 'scheduler'):
      trainer.scheduler.step(average_loss['total'])
    
    return average_loss, {
      'emotion_precision': emotion_precision,
      'sentiment_precision': sentiment_precision,
      'emotion_accuracy': emotion_accuracy,
      'sentiment_accuracy': sentiment_accuracy
    }

  # Main training loop with progress bar for epochs
  epoch_progress = tqdm(range(args.epochs), desc="Training progress")
  
  for epoch in epoch_progress:
    # Train with progress bar
    train_losses = train_epoch_with_progress(trainer)
    
    # Validate with progress bar
    dev_losses, dev_metrics = validate_with_progress(trainer, dev_loader)
    
    metrics_data["train_losses"].append(train_losses['total'])
    metrics_data["validation_losses"].append(dev_losses['total'])
    metrics_data["epochs"].append(epoch)
    
    # Update epoch progress bar with metrics
    epoch_progress.set_postfix({
      'train_loss': f"{train_losses['total']:.4f}",
      'val_loss': f"{dev_losses['total']:.4f}",
      'emo_acc': f"{dev_metrics['emotion_accuracy']:.2f}",
      'sent_acc': f"{dev_metrics['sentiment_accuracy']:.2f}"
    })
    
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
      epoch_progress.write(f"✓ Model saved at epoch {epoch+1} (val_loss: {dev_losses['total']:.4f})")
      
  # Evaluate on test set with progress bar
  test_losses, test_metrics = validate_with_progress(trainer, test_loader, "test")
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