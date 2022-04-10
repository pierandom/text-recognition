import os
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from datasets import get_dataset
from models import get_model
from trainer import Trainer
from loggers import get_logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['crnn'], default='crnn')
    parser.add_argument('--dataset', choices=['iiit5k'], default='iiit5k')
    parser.add_argument('--logger', choices=['wandb'], default=None)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run_id', type=str, help='run id to resume')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--base_lr', type=float, default=1e-1)
    parser.add_argument('--lr_base_period', type=int, default=15)
    parser.add_argument('--lr_period_factor', type=int, default=2)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--clip_grad_max_norm', type=float, default=1)
    parser.add_argument('--weight_decay_factor', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    return args


def main(args):
    if args.resume and not args.logger:
        print('Missing logger. Can\'t resume run')
        exit(1)
    
    if args.device == 'gpu' and not torch.cuda.is_available():
        print('GPU not available. Falling back on CPU')
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

    if args.logger:
        Logger = get_logger[args.logger]
    
    if args.resume:
        logger = Logger.previous_run(run_id=args.run_id, group=args.model)
        old_config = logger.get_config()
        args.__dict__.update(old_config)
    else:
        model_config_path = os.path.join('models', 'config', f'{args.model}.json')
        with open(model_config_path, 'r') as config_file:
            args.model_config = json.load(config_file)
        if args.logger:
            save_config_keys = {'base_lr', 'lr_base_period', 'lr_period_factor', 'weight_decay_factor'
                'clip_grad_max_norm', 'batch_size', 'momentum', 'model', 'model_config', 'dataset'}
            run_config = {key: value for key, value in vars(args).items() if key in save_config_keys}
            logger = Logger.new_run(config=run_config, group=args.model)

    # if resume, load dataset and model from previous run
    Dataset = get_dataset[args.dataset]
    Model = get_model[args.model]

    train_transforms = T.Compose([
        T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        T.ToTensor()])
    train_data = Dataset(split='train', transform=train_transforms)
    test_data = Dataset(split='test', transform=T.ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=test_data.collate_fn)
    
    model = Model(train_data.input_shape, train_data.num_classes, **args.model_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)
    scheduler1 = lr_scheduler.ConstantLR(optimizer, factor=1)
    scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_base_period, T_mult=args.lr_period_factor)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.warmup_epochs])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        logger=logger if args.logger else None,
        scheduler=scheduler,
        device=device,
        weight_decay_factor=args.weight_decay_factor,
        clip_grad_max_norm=args.clip_grad_max_norm)
    trainer.train(args.epochs)


if __name__ == '__main__':
    args = parse_args()
    main(args)