# Multimodal Emotion Recognition on MELD Dataset

A deep learning project that combines text, video, and audio features to recognize emotions in conversation.

## Overview

This project uses a multimodal approach to analyze emotional content in conversations from the MELD dataset, combining:

-   **Text**: BERT encoder for linguistic context
-   **Video**: 3D ResNet-18 for facial expressions and visual cues
-   **Audio**: Mel spectrograms for speech characteristics

## Setup Instructions

### Prerequisites

-   Python 3.12
-   PyTorch 1.9+
-   FFmpeg (required for audio/video processing)
-   CUDA-compatible GPU (recommended)

### Dataset Preparation

1. Download the MELD dataset:

    ```bash
    wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
    ```

2. Extract and organize the dataset:

    ```bash
    tar -xzvf MELD.Raw.tar.gz
    mkdir -p dataset/{train,dev,test}
    # Move files to their corresponding directories
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model Architecture

-   **Text Processing**: Fine-tuned BERT encoder extracts contextual embeddings from dialogue
-   **Video Processing**: 3D ResNet-18 captures temporal facial expressions and movements
-   **Audio Processing**: Mel spectrograms converted from raw audio represent speech patterns
-   **Fusion**: Late fusion combines features from all modalities for final emotion classification

## Usage

Train the model:

```bash
python training/meld_dataset.py
```

Evaluate on test set:

```bash
python training/eval.py --checkpoint path/to/model.pt
```

## License

[MIT License](LICENSE)
