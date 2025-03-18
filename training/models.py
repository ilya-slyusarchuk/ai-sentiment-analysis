import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch
from meld_dataset import MELD_Dataset
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from torch.amp import GradScaler

class TextEncoder(nn.Module):
    def __init__(self):
      super().__init__()
      self.bert = BertModel.from_pretrained('bert-base-uncased')
      
      for param in self.bert.parameters():
        param.requires_grad = False
      
      self.projection = nn.Linear(self.bert.config.hidden_size, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        
        pooler_output = outputs.pooler_output
        
        return self.projection(pooler_output)
      
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(weights=vision_models.video.R3D_18_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Gets the number of features from the last layer of the backbone to create a new linear layer with that number of input features
        num_features = self.backbone.fc.in_features
        # Replaces the last layer with a new linear layer with 128 output features, a ReLU activation function, and a dropout layer with a dropout rate of 0.2
        self.backbone.fc = nn.Sequential(
          nn.Linear(num_features, 128),
          nn.ReLU(),
          nn.Dropout(0.2)
        )

    def forward(self, video_frames):
        # Transpose so channels are the second dimension and the number of frames is the third dimension
        video_frames = video_frames.transpose(1, 2)
        # Pass through the backbone
        return self.backbone(video_frames)
      
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
          # Low-level features
          nn.Conv1d(64, 64, kernel_size=3),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2),
          # High-level features
          nn.Conv1d(64, 128, kernel_size=3),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.AdaptiveAvgPool1d(1)
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
          nn.Linear(128, 128),
          nn.ReLU(),
          nn.Dropout(0.2)
        )

    def forward(self, audio_data):
      # Squeeze the audio data to remove the channel dimension (mono audio)
      audio_data = audio_data.squeeze(1)
      # Pass through the conv layers
      outputs = self.conv_layers(audio_data)
      # Pass through the projection layer
      # Since we used AdaptiveAvgPool1d(1), the output is a 1D tensor with shape (batch_size, 128, 1)
      # We squeeze the last dimension to get a 1D tensor with shape (batch_size, 128)
      return self.projection(outputs.squeeze(-1))
    
class MultiModalTrainer:
    def __init__(self, model, train_loader, val_loader):
      self.model = model
      self.train_loader = train_loader
      self.val_loader = val_loader
      
      # Dataset sizes
      train_size = len(train_loader.dataset)
      val_size = len(val_loader.dataset)
      
      self.current_train_losses = None
      
      print("\nDataset sizes:")
      print(f"Train size: {train_size:,}")
      print(f"Validation size: {val_size:,}")
      print(f"Batches per epoch: {len(train_loader):,}")
      
      timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
      base_dir = f"opt/ml/output/tensorboard" if os.environ.get("SM_MODEL_DIR") else "runs"
      log_dir = os.path.join(base_dir, f"run_{timestamp}")
      self.writer = SummaryWriter(log_dir)
      self.global_step = 0
      
      self.optimizer = torch.optim.Adam([
        {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
        {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
        {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
        {'params': self.model.fusion_layer.parameters(), 'lr': 1e-4},
        {'params': self.model.emotion_classifier.parameters(), 'lr': 5e-4},
        {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4}
      ], weight_decay=1e-5)
      
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.1)
      
      self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
      self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
      
    def log_metrics(self, losses, metrics=None, phase="train"):
      if phase == "train":
        self.current_train_losses = losses
      else:
        self.writer.add_scalar("loss/total/train", self.current_train_losses['total'], self.global_step)
        self.writer.add_scalar("loss/total/validation", losses['total'], self.global_step)
        
        self.writer.add_scalar("loss/emotion/train", self.current_train_losses['emotions'], self.global_step)
        self.writer.add_scalar("loss/emotion/validation", losses['emotions'], self.global_step)
        
        self.writer.add_scalar("loss/sentiment/train", self.current_train_losses['sentiments'], self.global_step)
        self.writer.add_scalar("loss/sentiment/validation", losses['sentiments'], self.global_step)
        
        if metrics:
          self.writer.add_scalar(f"{phase}/emotion/precision", metrics['emotion_precision'], self.global_step)
          self.writer.add_scalar(f"{phase}/emotion/accuracy", metrics['emotion_accuracy'], self.global_step)
          
          self.writer.add_scalar(f"{phase}/sentiment/precision", metrics['sentiment_precision'], self.global_step)
          self.writer.add_scalar(f"{phase}/sentiment/accuracy", metrics['sentiment_accuracy'], self.global_step)
          
      
    def train_epoch(self):
      self.model.train()
      
      running_loss = {
        'total': 0.0,
        'emotions': 0.0,
        'sentiments': 0.0
      }
      
      # Get device
      device = next(self.model.parameters()).device
      device_type = device.type
      
      # Force model to use float32 for MPS compatibility
      if device_type == 'mps':
        for param in self.model.parameters():
          if param.data.dtype == torch.float64:
            param.data = param.data.float()  # Convert to float32
      
      # Use mixed precision for speedup - only for CUDA
      scaler = GradScaler() if device_type == 'cuda' else None
      
      # Enable mixed precision only for CUDA, not MPS
      use_amp = device_type == 'cuda'
      
      total_samples = 0
      
      for batch in self.train_loader:
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
        
        self.optimizer.zero_grad()
        
        # Use automatic mixed precision where supported
        if use_amp:
          with torch.autocast(device_type=device_type):
            outputs = self.model(text_inputs, video_frames, audio_data)
            emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
            loss = emotion_loss + sentiment_loss
            
          if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()
          else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
          outputs = self.model(text_inputs, video_frames, audio_data)
          emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
          sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
          loss = emotion_loss + sentiment_loss
          
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          self.optimizer.step()
        
        # Update running losses
        running_loss['total'] += loss.item() * batch_size
        running_loss['emotions'] += emotion_loss.item() * batch_size
        running_loss['sentiments'] += sentiment_loss.item() * batch_size
      
      # Calculate average losses
      for key in running_loss:
        running_loss[key] /= total_samples
        
      return running_loss
    
    def validate(self, data_loader, phase="validation"):
      self.model.eval()
      
      losses = {
        'total': 0.0,
        'emotions': 0.0,
        'sentiments': 0.0
      }
      
      # Get device 
      device = next(self.model.parameters()).device
      
      # Ensure float32 for MPS compatibility
      if device.type == 'mps':
        for param in self.model.parameters():
          if param.data.dtype == torch.float64:
            param.data = param.data.float()
      
      all_emotion_predictions = []
      all_sentiment_predictions = []
      all_emotion_labels = []
      all_sentiment_labels = []
      
      with torch.inference_mode():
        for batch in data_loader:
          text_inputs = {
            'input_ids': batch['text_input']['input_ids'].to(device),
            'attention_mask': batch['text_input']['attention_mask'].to(device)
          }
          video_frames = batch['video_frames'].to(device)
          audio_data = batch['audio_features'].to(device)
          emotion_labels = batch['emotion_labels'].to(device)
          sentiment_labels = batch['sentiment_labels'].to(device)
          
          outputs = self.model(text_inputs, video_frames, audio_data)
        
          emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
          sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
          
          loss = emotion_loss + sentiment_loss
          
          losses['total'] += loss.item()
          losses['emotions'] += emotion_loss.item()
          losses['sentiments'] += sentiment_loss.item()
          
          all_emotion_predictions.extend(outputs['emotions'].argmax(dim=1).cpu().numpy())
          all_sentiment_predictions.extend(outputs['sentiments'].argmax(dim=1).cpu().numpy())
          all_emotion_labels.extend(emotion_labels.cpu().numpy())
          all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
      
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
      
      self.log_metrics(average_loss, {
        'emotion_precision': emotion_precision,
        'sentiment_precision': sentiment_precision,
      }, phase)
      
      if phase == "validation":
        self.scheduler.step(average_loss['total'])
      
      return average_loss, {
        'emotion_precision': emotion_precision,
        'sentiment_precision': sentiment_precision,
        'emotion_accuracy': emotion_accuracy,
        'sentiment_accuracy': sentiment_accuracy
      }
          
class MultiModalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Set default tensor type to float32 to avoid MPS float64 issues
        torch.set_default_dtype(torch.float32)
        
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        self.fusion_layer = nn.Sequential(
          nn.Linear(128 * 3, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(0.2)
        )
        
        self.emotion_classifier = nn.Sequential(
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(64, 7)
        )
        
        self.sentiment_classifier = nn.Sequential(
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(64, 3)
        )
        
    def forward(self, text_inputs, video_data, audio_data):
      text_features = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])
      video_features = self.video_encoder(video_data)
      audio_features = self.audio_encoder(audio_data)
      
      # Concatenate the features. dim=1 means concatenate along the second dimension because the first dimension is the batch size
      combined_features = torch.cat((text_features, video_features, audio_features), dim=1)
      # Pass through the fusion layer
      fused_features = self.fusion_layer(combined_features)
      
      # Pass through the emotion classifier
      emotion_predictions = self.emotion_classifier(fused_features)
      # Pass through the sentiment classifier
      sentiment_predictions = self.sentiment_classifier(fused_features)
      
      return {
        'emotions': emotion_predictions,
        'sentiments': sentiment_predictions
      }