import os
import torch
from tqdm import tqdm
from torch.utils.data import random_split

import core
import datasets
import models
import core.losses
import core.optim
import core.datasets
from core.logger import Logger
from config import CONFIG_DEFAULT


# TODO: Change the type of DataLoader depending on your needs
from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset, DataLoader, random_split

VISUALIZATION_SIZE = 5
# TODO: Define your transforms here
TRANSFORMS = None

## TODO: This needs to modified for specific usecases
def train_one_epoch(model: torch.nn.Module,
                    loader: DataLoader, 
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    criterion,
                    device: str,
                    logger: Logger):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[Train] Epoch {epoch}")
    for step, batch in pbar:
        batch = batch.to(device)
        adj_hat, mu, logvar = model(batch.x, batch.edge_index, batch.batch)

        loss, recon_loss, kl_loss, _ = criterion(adj_hat, batch.edge_index, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        global_step = epoch * len(loader) + step
        logger.add_scalar("Train/Recon_Loss", recon_loss.item(), global_step)
        logger.add_scalar("Train/Total_Loss", loss.item(), global_step)
        logger.add_scalar("Train/KL_Loss", kl_loss.item(), global_step)

        pbar.set_postfix({"loss": loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()})

    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(loader)
    avg_kl = total_kl / len(loader)

    print(f"[Train] Epoch {epoch}: total={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
    return avg_loss

## TODO: This needs to modified for specific experiments 
@torch.no_grad()
def validate_one_epoch(model: torch.nn.Module,
                    loader: DataLoader, 
                    epoch: int,
                    criterion,
                    device: str,
                    logger: Logger):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0

    pbar = tqdm(loader, desc=f"[Val] Epoch {epoch}")
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            adj_hat, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
            loss, recon_loss, kl_loss, _ = criterion(adj_hat, batch.edge_index, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            pbar.set_postfix({"loss": loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()})

    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(loader)
    avg_kl = total_kl / len(loader)

    logger.add_scalar("Val/Total_Loss", avg_loss, epoch)
    logger.add_scalar("Val/Recon_Loss", avg_recon, epoch)
    logger.add_scalar("Val/KL_Loss", avg_kl, epoch)

    return avg_loss

def save_best_model(model, val_loss, best_val_loss, model_dir, exp_name):
    if val_loss < best_val_loss:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{exp_name}_best.pt")
        torch.save(model.state_dict(), path)
        print(f"Best model saved with val_loss={val_loss:.4f}")
        return val_loss
    return best_val_loss


# TODO: modify this depending on the experiment
def run_training(train_loader, validation_loader, visualization_loader,
                 epochs, model, optimizer, criterion, logger, cfg=None):
    
    device = cfg["experiment"]["device"]
    model_dir = cfg["experiment"]["model_dir"]
    exp_name = cfg["experiment"]["name"]
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, epoch, optimizer, criterion, device, logger)
        val_loss = validate_one_epoch(model, validation_loader, epoch, criterion, device, logger) \
            if validation_loader else train_loss
        # TODO: Implement a visualization logging during training 
        best_val_loss = save_best_model(model, val_loss, best_val_loss, model_dir, exp_name)

    print(f"Training finished. Best model in {model_dir}")


def handle_dataset_splits(train_dataset, test_dataset, cfg):
    loaders = {}
    try:
        is_random_state = cfg["data"]["random_state"]
        if is_random_state:
            g = torch.Generator()
            g.manual_seed(42)
        
        visualization_enabled = cfg["logging"]["visualization"]
        viz_size = 0

        if visualization_enabled:
            viz_size = VISUALIZATION_SIZE
        dataset_length = len(train_dataset) - viz_size
        train_size = int(0.9 * dataset_length)
        val_size = dataset_length - train_size

        train_dataset, viz_dataset, val_dataset = random_split(
            train_dataset, [train_size, viz_size, val_size], generator=g
        )

        train_batch_size = cfg["data"]["train_batch_size"]
        val_batch_size = cfg["data"]["val_batch_size"]
        test_batch_size = cfg["data"]["test_batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        viz_loader   = None if not visualization_enabled else DataLoader(viz_dataset, batch_size=1, shuffle=False)

        loaders["train_loader"] = train_loader
        loaders["viz_loader"] = viz_loader
        loaders["val_loader"] = val_loader
        loaders["test_loader"] = val_loader

    except Exception as e:
        print(f"Error: Splitting the dataset. Issue: {e}")
        exit(1)

    return loaders

def train():
    cfg = CONFIG_DEFAULT
    logger = Logger(cfg)

    # TODO: Implement a pre-trained loading feature
    # <=== load model[s] ===>
    model_cls = torch.models.get(cfg["model"]["arch"])
    model = model_cls(**cfg["model"]["model_params"]).to(cfg["experiment"]["device"])

    # <==== load optimizer[s] ===>
    optim_fn = torch.optimizers.get(cfg["train"]["optimizer"])
    optimizer = optim_fn(model.parameters(), cfg)
    
    # <===== load loss[es] ====>
    loss_cls = torch.losses.get(cfg["train"]["loss"])
    criterion = loss_cls

    # <=== load dataset[s] ====>
    dataset_params = cfg["data"]["dataset_params"]
    dataset_cls = torch.datasets.get(cfg["data"]["dataset"])

    # <=== load TRAIN dataset ====>
    train_dataset = dataset_cls(transforms=TRANSFORMS, **dataset_params)

    # <=== load TEST dataset ====>
    test_dataset = dataset_cls(train=False, **dataset_params)

    # <===== handle train-test-val-visualiation split ===>
    loaders = handle_dataset_splits(train_dataset, test_dataset, cfg)

    train_loader = loaders.get("train_loader", None)
    val_loader = loaders.get("val_loader", None)
    viz_loader = loaders.get("viz_loader", None)
    # test_loader = loaders.get("test_loader", None)

    epochs = cfg["train"]["epochs"]

    try:
        # Check if everyone has been loaded properly
        assert train_dataset is not None
        assert val_loader is not None
        assert epochs is not None
        assert model is not None
        assert optimizer is not None
        assert criterion is not None
        assert logger is not None

        run_training(
                    train_loader=train_loader,
                    validation_loader=val_loader, 
                    visualization_loader=viz_loader, 
                    epochs=epochs, 
                    model=model,
                    optimizer=optimizer, 
                    criterion=criterion,
                    logger=logger,
                    cfg=cfg
        )
    except Exception as e:
        print(f"Error: During Training, Issue: {e}")
        exit(1)

    logger.close()

if __name__ == "__main__":
    train()
