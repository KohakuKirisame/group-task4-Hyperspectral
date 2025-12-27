import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time

class CustomTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 device='cuda',
                 criterion=nn.MSELoss(),
                 log_interval=100,
                 exp_name=None,
                 log_dir='./logs',
                 scheduler=None,):

        self.optimizer = optimizer
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.log_interval = log_interval
        self.scheduler = scheduler

        if exp_name is None:
            exp_name = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, exp_name)
        )
        self.global_step = 0

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for step, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x, target = batch
            else:
                x = batch
                target = batch

            x = x.to(self.device)
            target = target.to(self.device)

            x, x_hat, z = self.model(x)
            loss = self.criterion(x_hat, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar("loss/train_step",
                                       loss.item(),
                                       global_step=self.global_step)
                pbar.set_postfix(loss=loss.item())
            self.global_step += 1
            self.scheduler.step()


        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, target = batch
            else:
                x = batch
                target = batch
            x = x.to(self.device)
            target = target.to(self.device)

            x, x_hat, z = self.model(x)
            loss = self.criterion(x_hat, target)
            total_loss += loss.item()

        self.writer.add_scalar("loss/val", total_loss / len(dataloader), epoch)
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader=None, epochs=4):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)

            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"Train Loss: {train_loss:.6f}"
                )

            torch.save(self.model.state_dict(), './resources/models/model_' + str(epoch) + '.pth')


class CustomClassifierTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 device='cuda',
                 criterion=nn.CrossEntropyLoss(label_smoothing=0.0),
                 log_interval=100,
                 exp_name=None,
                 log_dir='./logs/classifier/',
                 scheduler=None,
                 save_interval=5000,):
        self.optimizer = optimizer
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.log_interval = log_interval
        self.scheduler = scheduler
        self.reconstruction_loss = nn.MSELoss()
        self.save_interval = save_interval

        if exp_name is None:
            exp_name = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, exp_name, 'classifier')
        )
        self.global_step = 0

    def train_epoch(self, dataloader, epoch, val_loader=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            logits, x_hat = self.model(x)
            entropy_loss = self.criterion(logits, y)
            reconstruction_loss = self.reconstruction_loss(x_hat, x)

            loss = entropy_loss + 0.2 * reconstruction_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            acc = correct / total
            pbar.set_postfix(loss=loss.item(), acc=acc)

            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar("cls/train_loss", loss.item(), self.global_step)
                self.writer.add_scalar("cls/train_acc", acc, self.global_step)
                self.writer.add_scalar("cls/train_entropy", entropy_loss.item(), self.global_step)
                self.writer.add_scalar("cls/train_reconstruction", reconstruction_loss.item(), self.global_step)

            if self.global_step % self.save_interval == 0:
                torch.save(self.model.state_dict(), './resources/models/classifier_model_' + str(self.global_step) + '.pth')
            if val_loader is not None and self.global_step % (self.save_interval // 2) == 0:
                val_loss, val_acc  = self.eval_epoch(val_loader, epoch)
                self.model.train()
                print(
                    f"[Epoch {self.global_step:03d}] "
                    f"Train Loss: {loss:.6f} | "
                    f"Train Acc: {acc:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val Acc: {val_acc:.6f} | "
                )
                self.writer.add_scalar("cls/val_loss", val_loss, self.global_step)
                self.writer.add_scalar("cls/val_acc", val_acc, self.global_step)
            self.global_step += 1
            self.scheduler.step()

        return total_loss / len(dataloader), acc

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits, x_hat = self.model(x)
            entropy_loss = self.criterion(logits, y)
            reconstruction_loss = self.reconstruction_loss(x_hat, x)

            loss = entropy_loss + 0.2 * reconstruction_loss

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / len(dataloader)

        # self.writer.add_scalar("cls/val_loss", avg_loss, epoch)
        # self.writer.add_scalar("cls/val_acc", acc, epoch)

        return avg_loss, acc

    def train(self, train_loader, val_loader=None, epochs=4):
        for epoch in range(1, epochs + 1):
            train_loss, acc  = self.train_epoch(train_loader, epoch, val_loader)

            if val_loader is not None:
                val_loss, val_acc  = self.eval_epoch(val_loader, epoch)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Train Acc: {acc:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val Acc: {val_acc:.6f} | "
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"Train Loss: {train_loss:.6f}"
                )

            # torch.save(self.model.state_dict(), './resources/models/classifier_model_' + str(epoch) + '.pth')
