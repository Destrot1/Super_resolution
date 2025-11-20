import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision


logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, device, num_epochs, lr, save_path):

    writer = SummaryWriter(log_dir="runs/experiment1")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    ) # learning_rate scheduler

    for epoch in range(num_epochs):
        model.train()
        train_loss_accumulator = 0.0
        
        for ivlr_patch, lr_patch in train_loader:
            ivlr_patch = ivlr_patch.to(device)
            lr_patch = lr_patch.to(device)

            optimizer.zero_grad()
            output = model(ivlr_patch)
            loss = criterion(output, lr_patch)
            loss.backward()
            optimizer.step()

            train_loss_accumulator += loss.item()
        
        train_loss = train_loss_accumulator / len(train_loader) # mean_batch_loss

        model.eval()
        val_loss_accumulator = 0.0

        with torch.no_grad():
            for ivlr_patch, lr_patch in val_loader:
                ivlr_patch = ivlr_patch.to(device)
                lr_patch = lr_patch.to(device)

                output = model(ivlr_patch)
                loss = criterion(output, lr_patch)
                val_loss_accumulator += loss.item()

        val_loss = val_loss_accumulator/len(val_loader)
        scheduler.step(val_loss) 


        # tensorboard log
        # Loss
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Learning Rate
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # images: input vs output vs target
        # first validation batch
        iv_sample, target_sample = next(iter(val_loader))
        iv_sample = iv_sample[:4].to(device)         # first 4 examples
        target_sample = target_sample[:4].to(device)
        output_sample = model(iv_sample)

        # create visualization grid
        grid_input  = torchvision.utils.make_grid(iv_sample.cpu(), nrow=4, normalize=True)
        grid_output = torchvision.utils.make_grid(output_sample.cpu(), nrow=4, normalize=True)
        grid_target = torchvision.utils.make_grid(target_sample.cpu(), nrow=4, normalize=True)

        writer.add_image("Input (IVLR)",  grid_input,  epoch)
        writer.add_image("Output",        grid_output, epoch)
        writer.add_image("Target",        grid_target, epoch)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] " f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved in: {save_path}")

    writer.close()
    return model






