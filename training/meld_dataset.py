from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class MELD_Dataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {path}")
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read video from {path}")
            
            cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            
            # Max 30 frames
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize the frame to 224x224 and normalize RGB values to [0, 1]
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                
                # Add the frame to the list
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Error loading video frames from {path}: {e}")
            
        finally:
            cap.release()
            
        # If no frames were found, raise an error
        if (len(frames) == 0):
            raise ValueError(f"No frames found in video {path}")
        elif (len(frames) < 30):
            # Pad the frames with the last frame
            while len(frames) < 30:
                frames.append(np.zeros_like(frames[0]))
        elif (len(frames) > 30):
            # Trim the frames to 30
            frames = frames[:30]
        
        # Convert the frames to a PyTorch tensor and permute the dimensions to (30, 3, 224, 224) [Frames, Channels, Height, Width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, path):
        audio_path = path.replace('.mp4', '.wav')
        
        try:
            subprocess.run([
                'ffmpeg', 
                '-i', path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000', 
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            waveform, sr = torchaudio.load(audio_path)
            
            if (sr != 16000):
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            mel_spectogram = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=512, n_mels=64)(waveform)
            
            mel_spectogram = (mel_spectogram - mel_spectogram.mean()) / mel_spectogram.std()
            
            if (mel_spectogram.size(2) < 300):
                padding = 300 - mel_spectogram.size(2)
                mel_spectogram = torch.nn.functional.pad(mel_spectogram, (0, padding))
            elif mel_spectogram.size(2):
                mel_spectogram = mel_spectogram[:, :, :300]
                
            return mel_spectogram
                
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio from {path}: {e}")
        except Exception as e:
            raise ValueError(f"Audio error: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
            
        row = self.data.iloc[index]
        try:
            video_file = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_file)
            video_path = os.path.exists(path)
            
            if not video_path:
                raise FileNotFoundError(f"Video file {path} does not exist")
            
            # Tokenize the text, converts to PyTorch tensors and pad to max length of 128
            text_input = self.tokenizer(row['Utterance'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)
            
            # Map the emotion and sentiment to their respective indices
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_input': {
                    'input_ids': text_input['input_ids'].squeeze(),
                    'attention_mask': text_input['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_labels': torch.tensor(emotion_label),
                'sentiment_labels': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None
        
def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    # Use PyTorch's default collate function but ensure float32 for numerical tensors
    batch_collated = torch.utils.data.dataloader.default_collate(batch)
    
    # Ensure float32 for video and audio tensors
    if 'video_frames' in batch_collated:
        batch_collated['video_frames'] = batch_collated['video_frames'].float()
    
    if 'audio_features' in batch_collated:
        batch_collated['audio_features'] = batch_collated['audio_features'].float()
    
    return batch_collated
    
def prepare_dataloader(train_csv_path, train_video_dir, dev_csv_path, dev_video_dir, test_csv_path, test_video_dir, batch_size=32):
    train_dataset = MELD_Dataset(train_csv_path, train_video_dir)
    dev_dataset = MELD_Dataset(dev_csv_path, dev_video_dir)
    test_dataset = MELD_Dataset(test_csv_path, test_video_dir)
    
    # Determine optimal number of workers
    # Use half the CPU cores, capped at 8 to avoid overhead
    print(f"CPU cores: {os.cpu_count()}")
    num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 2
    print(f"Using {num_workers} dataloader workers")
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    return train_loader, dev_loader, test_loader