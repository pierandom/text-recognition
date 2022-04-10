import json
import os
import wandb
import torch


class WandbLogger:
    def __init__(self, run):
        self.run = run
        self.config_path = os.path.join(self.run.dir, 'config.json')
        self.checkpoint_path = os.path.join(self.run.dir, 'checkpoint.pth')

    @classmethod
    def new_run(cls, config, group):
        run = wandb.init(
            project='text-recognition',
            group=group,
            config = config)
        logger = cls(run)
        logger.resume = False
        logger.save_config(config)
        return logger
    
    @classmethod
    def previous_run(cls, run_id, group):
        run = wandb.init(
            project='text-recognition',
            group=group,
            resume='must',
            id=run_id)
        logger = cls(run)
        logger.resume = True
        return logger
    
    def log(self, stats, step):
        self.run.log(stats, step)
    
    def get_config(self):
        self.run.restore('config.json')
        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    
    def save_config(self, config):
        with open(self.config_path, 'w') as config_file:
            json.dump(config, config_file)
        self.run.save(self.config_path, base_path=wandb.run.dir)
    
    def get_checkpoint(self):
        self.run.restore('checkpoint.pth')
        checkpoint = torch.load(self.checkpoint_path)
        return checkpoint
    
    def save_checkpoint(self, checkpoint):
        torch.save(checkpoint, self.checkpoint_path)
        self.run.save(self.checkpoint_path, base_path=wandb.run.dir)