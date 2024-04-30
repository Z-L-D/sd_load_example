import torch
from omegaconf import OmegaConf
import pathlib
from ldm.util import instantiate_from_config
import safetensors.torch as st_torch
import pytorch_lightning as pl


print("PyTorch version:", torch.__version__)
print("PyTorch Lightning version:", pl.__version__)

def load_model_from_config(config_path, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    with open(ckpt, 'rb') as f:  # Open the file in binary mode
        file_bytes = f.read()  # Read the entire file content into bytes
        pl_sd = st_torch.load(file_bytes)  # Load from bytes

    # print("Loaded keys:", pl_sd.keys())  # Inspect keys in the loaded dictionary

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    if 'state_dict' in pl_sd:
        sd = {k: torch.as_tensor(v) for k, v in pl_sd["state_dict"].items()}  # Convert tensors manually if necessary
    else:
        # If 'state_dict' is not a key, assume the entire dictionary is the state_dict
        sd = {k: torch.as_tensor(v) for k, v in pl_sd.items()}

    # Load the configuration file
    config = OmegaConf.load(config_path)  # Assuming the configuration path is correct

    # Create the model from the configuration
    model = instantiate_from_config(config.model)

    # print("model", model)

    # Access the VAE
    vae = model.first_stage_model
    
    # Access the U-Net
    unet = model.model.diffusion_model

    print("\n============================\n")
    print("vae: ", vae)
    print("\n============================\n")
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    print("\n============================\n")
    print("unet: ", unet)
    print("\n============================\n")

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()  # Uncomment this line if you are running on a machine with a GPU
    model.eval()
    return model

config_path = ""
ckpt = ""

load_model_from_config(config_path, ckpt, verbose=False)
