import argparse
import os

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import trange

from src import utils
from data.data import get_train_data, get_train_augments
from src.models.resnet20 import ResNet20


def train_epoch(model, optimizer, criterion, loader, scheduler=None, device="cpu", mixup_alpha=.2):
    model.train()
    losses = torch.zeros((1,), device=device)
    scaler = torch.cuda.amp.GradScaler()
    for batch_id, (input, target) in enumerate(loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        input, target_a, target_b, lmbd = utils.mixup_data(input, target, mixup_alpha, device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            output = model(input)
            loss = utils.mixup_criterion(criterion, output, target_a, target_b, lmbd)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        losses += loss.detach()

    return losses / len(loader)


@torch.no_grad()
def evaluate(model, criterion, loader, device="cpu", num_batches=None, batch_preprocessor=lambda x: x):
    model.eval()
    accuracy = 0
    val_loss = torch.zeros(1, device=device, dtype=torch.float32)
    for batch_id, (input, target) in enumerate(loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        input = batch_preprocessor(input)
        output = model(input)
        loss = criterion(output, target)

        val_loss += loss.detach()
        accuracy += (output.argmax(dim=1) == target).float().mean()

        if num_batches is not None and batch_id > num_batches:
            break
    return val_loss / len(loader), accuracy / len(loader)


def train(model, optimizer, criterion, train_loader, test_loader, num_epochs, scheduler=None, device="cpu",
          mixup_alpha=.2):
    model.to(device)
    best_accuracy = 0

    for epoch in trange(1, num_epochs + 1):
        epoch_loss = train_epoch(model, optimizer, criterion, train_loader, scheduler=scheduler,
                                 device=device, mixup_alpha=mixup_alpha)
        val_loss, val_accuracy = evaluate(model, criterion, test_loader, device=device)

        wandb.log({
            "train_loss": epoch_loss.cpu(),
            "val_loss": val_loss.cpu(),
            "lr": optimizer.param_groups[0]["lr"],
            "accuracy": val_accuracy.cpu()
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join("checkpoints", f"resnet20.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, help="MixUp coeff", default=.8)
    parser.add_argument('--lr', type=float, help="learning rate", default=4e-3)
    parser.add_argument('--wd', type=float, help="weight decay", default=5e-2)
    parser.add_argument('--seed', type=int, help="Random seed", default=322)
    parser.add_argument('--nw', type=int, help="Num workers", default=6)
    parser.add_argument('--bs', type=int, help="Batch size", default=128)
    parser.add_argument('--epochs', type=int, help="Number of epoches to train", default=50)
    parser.add_argument('--num_augments', type=int, default=2)
    parser.add_argument('--magnitude', type=int, help="RandAugment magnitude", default=14)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default="cuda")
    args = dict(vars(parser.parse_args()))
    seed, num_workers, device = args["seed"], args["nw"], args["device"]

    utils.set_random_seed(seed)

    train_augs = get_train_augments(args["num_augments"], args["magnitude"])
    cifar10_train, cifar10_test = get_train_data(root_dir="data", train_transforms=train_augs)
    train_loader = DataLoader(cifar10_train, batch_size=args["bs"], shuffle=True,
                              pin_memory=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(cifar10_test, batch_size=args["bs"], shuffle=False,
                             pin_memory=True, num_workers=num_workers)

    num_train_steps = len(train_loader) * args["epochs"] + 2
    num_warmup_steps = len(train_loader) * args["warmup_epochs"] + 2
    model = ResNet20(configuration=(3, 2, 2), num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["wd"],
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=num_train_steps,
                                                    pct_start=num_warmup_steps / num_train_steps, anneal_strategy="cos",
                                                    max_lr=args["lr"], div_factor=100)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    with wandb.init(project="quatization_simple", entity="johan_ddc_team", name="resnet20_train") as run:
        wandb.watch(model, optimizer, log="all", log_freq=10)
        train(model, optimizer, criterion, train_loader, test_loader, num_epochs=args["epochs"],
              scheduler=scheduler, device=device, mixup_alpha=args["alpha"])
        torch.save(model.state_dict(), os.path.join("checkpoints", f"resnet20_final.pth"))
