import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ai_edge_torch
import os
from contextlib import contextmanager, nullcontext
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large")

# Define input shapes
batch_size = 1
sequence_length = 80000  # Approximately 5 seconds of audio at 16kHz

# Create dummy input tensor
dummy_input = (torch.randn(batch_size, sequence_length),)


encoder = asr_model.encoder

print(encoder)

encoder.export("encoder.tflite")

exit()

@contextmanager
def monkeypatched(object, name, patch):
    """Temporarily monkeypatches an object."""
    pre_patched_value = getattr(object, name)
    setattr(object, name, patch)
    yield object
    setattr(object, name, pre_patched_value)

# Set module mode
with torch.inference_mode(), torch.no_grad(), torch.jit.optimized_execution(True):
    with monkeypatched(torch.nn.RNNBase, "flatten_parameters", lambda *args: None):
        edge_model = ai_edge_torch.convert(encoder.eval(), dummy_input)
        edge_model.export("encoder.tflite")
