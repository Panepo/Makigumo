import torch
from collections import namedtuple
from PhotoMaker.photomaker.model import PhotoMakerIDEncoder

class OVIDEncoderWrapper(PhotoMakerIDEncoder):
    dtype = torch.float32  # accessed in the original workflow

    def __init__(self, id_encoder, orig_id_encoder):
        super().__init__()
        self.id_encoder = id_encoder
        self.modules = orig_id_encoder.modules  # accessed in the original workflow
        self.config = orig_id_encoder.config  # accessed in the original workflow

    def __call__(
        self,
        *args,
    ):
        id_pixel_values, prompt_embeds, class_tokens_mask = args
        inputs = {
            "id_pixel_values": id_pixel_values,
            "prompt_embeds": prompt_embeds,
            "class_tokens_mask": class_tokens_mask,
        }
        output = self.id_encoder(inputs)[0]
        return torch.from_numpy(output)

class OVTextEncoderWrapper:
    dtype = torch.float32  # accessed in the original workflow

    def __init__(self, text_encoder, orig_text_encoder):
        self.text_encoder = text_encoder
        self.modules = orig_text_encoder.modules  # accessed in the original workflow
        self.config = orig_text_encoder.config  # accessed in the original workflow

    def __call__(self, input_ids, **kwargs):
        inputs = {"input_ids": input_ids}
        output = self.text_encoder(inputs)

        hidden_states = []
        hidden_states_len = len(output)
        for i in range(1, hidden_states_len):
            hidden_states.append(torch.from_numpy(output[i]))

        BaseModelOutputWithPooling = namedtuple("BaseModelOutputWithPooling", "last_hidden_state hidden_states")
        output = BaseModelOutputWithPooling(torch.from_numpy(output[0]), hidden_states)
        return output

class OVUnetWrapper:
    def __init__(self, unet, unet_orig):
        self.unet = unet
        self.config = unet_orig.config  # accessed in the original workflow
        self.add_embedding = unet_orig.add_embedding  # accessed in the original workflow

    def __call__(self, *args, **kwargs):
        latent_model_input, t = args
        inputs = {
            "sample": latent_model_input,
            "timestep": t,
            "encoder_hidden_states": kwargs["encoder_hidden_states"],
            "text_embeds": kwargs["added_cond_kwargs"]["text_embeds"],
            "time_ids": kwargs["added_cond_kwargs"]["time_ids"],
        }

        output = self.unet(inputs)

        return [torch.from_numpy(output[0])]

class OVVAEDecoderWrapper:
    dtype = torch.float32  # accessed in the original workflow

    def __init__(self, vae, vae_orig):
        self.vae = vae
        self.config = vae_orig.config  # accessed in the original workflow

    def decode(self, latents, return_dict=False):
        output = self.vae(latents)[0]
        output = torch.from_numpy(output)

        return [output]
