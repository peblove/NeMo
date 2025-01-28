# Import NeMo ASR module
import nemo.collections.asr as nemo_asr

# Load FastConformer CTC large speech recognition model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_fastconformer_ctc_large")

# Extract encoder and export to TFLite format
encoder = asr_model.encoder
encoder.export("encoder.tflite")

# Extract decoder and export to TFLite format
decoder = asr_model.decoder
decoder.export("decoder.tflite")
