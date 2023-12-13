import platform
import torch

def systemCheck():
    if (platform.system() == "Darwin"):
        print("Using MacOS!")
        if torch.backends.mps.is_built():
            print("There is a GPU available.")
        else:
            print("There is no GPU available.")
        return "mps"
    else:
        print("Using Windows!")
        if torch.cuda.is_available():
            print("There is a GPU available.")
        else:
            print("There is no GPU available.")
        return "cuda"
