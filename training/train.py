import torch

def train(optimizer, criterion, model, dataloader, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):

        # Just to check for testing:
        # Must be shape of (B, 3, 100, 100) for HR and shape (B, 3, 48, 48) for LR
        if epoch == 0:
            print("Output shape:", outputs.shape)
            print("Target shape:", IH.shape)
        
        epoch_loss = 0.0
        # Iin: low-res input, IH: high-res target
        for Iin, IH in dataloader:
            Iin, IH = Iin.to(device).float(), IH.to(device).float()

            # zero the parameter gradients dont want to accumulate gradients across batches
            optimizer.zero_grad()

            # forward pass
            outputs = model(Iin)

            # compute loss
            loss = criterion(outputs, IH)
            epoch_loss += loss.item()

            # backward pass and optimization
            loss.backward()

            # I added gradient clipping, can remove if paper doesn't mention it, but it can help stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        optimizer.zero_grad()
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")