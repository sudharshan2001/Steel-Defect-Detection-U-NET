from tqdm import tqdm
import torch
from info import CFG

DEVICE = CFG.DEVICE

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)

        targets = targets.float().to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        loop.set_postfix(loss=loss.item())