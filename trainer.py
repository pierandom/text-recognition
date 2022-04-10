from datetime import timedelta
import time
import torch

from metrics import Mean, WordAccuracy

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader=None,
                 logger=None, scheduler=None, device=torch.device('cpu'),
                 weight_decay_factor=None, clip_grad_max_norm=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ctc_loss = torch.nn.CTCLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.weight_decay_factor = weight_decay_factor
        self.clip_grad_max_norm = clip_grad_max_norm
        self.logger = logger
        self.start_epoch = 0
        if logger and logger.resume:
            self._load_checkpoint()
    
    def _load_checkpoint(self):
        checkpoint = self.logger.get_checkpoint()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']

    def train_step(self):
        self.model.train()
        train_loss = Mean()
        train_accuracy = WordAccuracy()

        for image, target in self.train_loader:
            image_data = image['data'].to(self.device)
            image_width = image['width'].to(self.device)
            target_length = target['length'].to(self.device)
            encoded_labels = target['encoded_label'].to(self.device)

            logits = self.model(image_data)
            input_length = (image_width / self.model.reduction_factor).ceil().to(torch.int32)
            loss = self.ctc_loss(logits.log_softmax(dim=-1), encoded_labels, input_length, target_length)
            if self.weight_decay_factor:
                loss += self.weight_decay_factor * self.model.weight_decay(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_max_norm)
            self.optimizer.step()

            train_loss.update(loss.item())
            predicted_labels = self.train_loader.dataset.decode_predictions(logits.detach().cpu())
            train_accuracy.update(predicted_labels, target['label'])
        
        stats = {
            'train_loss': train_loss.compute(),
            'train_accuracy': train_accuracy.compute()}
        return stats


    def evaluate(self):
        self.model.eval()
        val_loss = Mean()
        val_accuracy = WordAccuracy()

        with torch.inference_mode():
            for image, target in self.test_loader:
                image_data = image['data'].to(self.device)
                image_width = image['width'].to(self.device)
                target_length = target['length'].to(self.device)
                encoded_labels = target['encoded_label'].to(self.device)

                logits = self.model(image_data)
                input_length = (image_width / self.model.reduction_factor).ceil().to(torch.int32)
                loss = self.ctc_loss(logits.log_softmax(dim=-1), encoded_labels, input_length, target_length)

                val_loss.update(loss.item())
                predicted_labels = self.test_loader.dataset.decode_predictions(logits.cpu())
                val_accuracy.update(predicted_labels, target['label'])          
        
        stats = {
            'val_loss': val_loss.compute(),
            'val_accuracy': val_accuracy.compute()}
        return stats


    def train(self, epochs):
        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f'Model: {self.model.__class__.__name__} - Parameters: {num_parameters:,}')
        print(f'Training on {"GPU" if self.device.type == "cuda" else "CPU"}...')
        tot_epochs = self.start_epoch + epochs
        epoch_fmt = f'>{len(str(tot_epochs))}'
        train_start_time = time.time()
        for epoch in range(self.start_epoch+1, tot_epochs+1):
            epoch_start_time = time.time()
            stats = self.train_step()
            if self.test_loader:
                stats |= self.evaluate()
            if self.scheduler:
                self.scheduler.step()
            epoch_time = int(time.time() - epoch_start_time)
            train_time = int(time.time() - train_start_time)
            if self.logger:
                self.logger.log(stats, step=epoch)
                checkpoint = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}
                if self.scheduler:
                    checkpoint |= {'scheduler': self.scheduler.state_dict()}
                self.logger.save_checkpoint(checkpoint)
            epoch_summary = (
                f'Epoch: {epoch:{epoch_fmt}}/{tot_epochs} - '
                f'Time: {timedelta(seconds=epoch_time)}/{timedelta(seconds=train_time)} - '
                f'Train Loss: {stats["train_loss"]:.6f} - Train Accuracy: {stats["train_accuracy"]:.2%} - ')
            if self.test_loader:
                epoch_summary += (
                    f'Val Loss: {stats["val_loss"]:.6f} - Val Accuracy: {stats["val_accuracy"]:.2%}')
            print(epoch_summary)
