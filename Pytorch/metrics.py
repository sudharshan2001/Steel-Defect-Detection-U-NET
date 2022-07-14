import torch


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            dice_score += (2 * (preds * y).sum()+1e-6) / (
                (preds + y).sum() + 1e-6
            )
            

    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
