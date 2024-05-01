import os
import sys
import torch
from diffusers import DiffusionPipeline

torch_device = "cuda"

pipe = DiffusionPipeline.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--digiplay--Juggernaut_final_full").to("cuda")

# Print the model structure within the pipeline
# print("\n============================\n")
# print("Model Structure:", pipe.model)
# print("\n============================\n")

print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

print("\n============================\n")
print("Model Structure:", pipe.unet)
print("\n============================\n")

print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

print("\n============================\n")
print("Model Structure:", pipe.vae)
print("\n============================\n")

print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

