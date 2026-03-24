import torch
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        # As a reminder =  Iin: low-res input, IH: high-res target
        for Iin, IH in dataloader:
            Iin, IH = Iin.to(device), IH.to(device)

            # forward pass
            outputs = model(Iin)
            # We have to compute metrics here, but for now we just print the shapes
            print(f"Input (LR) shape: {Iin.shape}, Output (SR) shape: {outputs.shape}, Target (HR) shape: {IH.shape}")