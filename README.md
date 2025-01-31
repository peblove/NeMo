# NeMo (PyTorch TorchScript) models to LiteRT (formerly TFLite) Direct Conversion

This repository demonstrates how to directly convert NeMo models to LiteRT (formerly TFLite) int8 quantized models without using ONNX as an intermediate step.

## Environment Prerequisites

Due to compatibility requirements:
- Python 3.10 is recommended (ai_edge_torch does not currently support Python 3.12)
- CUDA-compatible environment

## Installation

### 1. System Dependencies
Install required system packages:
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
```

### 2. TensorFlow Installation
Install TensorFlow with CUDA support first due to version compatibility with PyTorch 2.5.1:
```bash
pip install tensorflow[and-cuda]==2.18.0
```

### 3. Additional Dependencies
Install remaining required packages:
```bash
pip install -r requirements.txt
```

## Model Conversion

To convert the `stt_en_fastconformer_ctc_large` model to LiteRT format:
```bash
python nemo2litert.py
```

## Additional Features

- Supports conversion of cache-aware streaming models
- Compatible with dynamic axes models
- Enables direct int8 quantization

## Technical Notes

- The conversion process bypasses ONNX intermediates for more efficient transformation
- Supports various NeMo model architectures
- Maintains model performance while reducing size through int8 quantization
