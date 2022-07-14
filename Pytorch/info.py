import torch
class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-6
    BATCH_SIZE = 16
    LOAD_MODEL = False
    EPOCH = 10
    SAVE_PATH = "./checkpoint/my_checkpoint1.pth.tar"

