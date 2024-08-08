import os
from pathlib import Path
from PhotoMaker.photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
import gc
import openvino as ov
from pathlib import Path
from ovwrapper import OVIDEncoderWrapper, OVTextEncoderWrapper, OVUnetWrapper, OVVAEDecoderWrapper

trigger_word = "img"

photomaker_path = Path("models/photomaker-v1.bin")
base_model_path = "models/RealVisXL_V3.0"

def load_original_pytorch_pipeline_components(photomaker_path: str, base_model_id: str):
    # Load base model
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(base_model_id, use_safetensors=True).to("cpu")

    # Load PhotoMaker checkpoint
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
        trigger_word=trigger_word,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    gc.collect()
    return pipe

pipe = load_original_pytorch_pipeline_components(photomaker_path, base_model_path)

TEXT_ENCODER_OV_PATH = Path("models/text_encoder.xml")
TEXT_ENCODER_2_OV_PATH = Path("models/text_encoder_2.xml")
UNET_OV_PATH = Path("models/unet.xml")
ID_ENCODER_OV_PATH = Path("models/id_encoder.xml")
VAE_DECODER_OV_PATH = Path("models/vae_decoder.xml")

core = ov.Core()
device = 'GPU'

compiled_id_encoder = core.compile_model(ID_ENCODER_OV_PATH, device)
compiled_unet = core.compile_model(UNET_OV_PATH, device)
compiled_text_encoder = core.compile_model(TEXT_ENCODER_OV_PATH, device)
compiled_text_encoder_2 = core.compile_model(TEXT_ENCODER_2_OV_PATH, device)
compiled_vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device)

pipe.id_encoder = OVIDEncoderWrapper(compiled_id_encoder, pipe.id_encoder)
pipe.unet = OVUnetWrapper(compiled_unet, pipe.unet)
pipe.text_encoder = OVTextEncoderWrapper(compiled_text_encoder, pipe.text_encoder)
pipe.text_encoder_2 = OVTextEncoderWrapper(compiled_text_encoder_2, pipe.text_encoder_2)
pipe.vae = OVVAEDecoderWrapper(compiled_vae_decoder, pipe.vae)
