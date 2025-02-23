# Import NVIDIA's NeMo ASR (Automatic Speech Recognition) toolkit, which provides 
# state-of-the-art speech recognition models and components
import nemo.collections.asr as nemo_asr

# Load the FastConformer CTC large model - a high-accuracy English ASR model
# that uses the Conformer architecture with CTC loss and BPE tokenization
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large")

# Extract the encoder (acoustic model) which converts audio features into 
# high-level representations
encoder = asr_model.encoder

# Export the encoder to TFLite format for mobile/edge deployment
# dynamic_axes={} ensures fixed input dimensions for better optimization
encoder.export("stt_en_fastconformer_ctc_large_encoder.tflite", dynamic_axes={})

# Extract the decoder which converts the encoder's representations into 
# text predictions
decoder = asr_model.decoder

# Export the decoder to TFLite format for mobile/edge deployment
# dynamic_axes={} ensures fixed input dimensions for better optimization
decoder.export("stt_en_fastconformer_ctc_large_decoder.tflite", dynamic_axes={})

# Load the QuartzNet 15x5 model trained by RobotsMali
# This is a lightweight ASR model with 15 jasper blocks and 5 sub-blocks
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="RobotsMali/stt-bm-quartznet15x5")

# Extract the QuartzNet encoder which processes the raw audio input
encoder = asr_model.encoder

# Export the encoder to TFLite format with fixed dimensions
# This makes the model more efficient for deployment
encoder.export("RobotsMali_encoder.tflite", dynamic_axes={})

# Extract the QuartzNet decoder which generates the final text output
decoder = asr_model.decoder

# Export the decoder to TFLite format with fixed dimensions
# This ensures consistent performance across different platforms
decoder.export("RobotsMali_decoder.tflite", dynamic_axes={})
