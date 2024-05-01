import os
import argparse

print(os.system('nvidia-smi'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train import trainer
from utils import set_seed

parser = argparse.ArgumentParser(description='Audio Classification')

parser.add_argument('--dataset', type=str, default='urbansound', help='Dataset to use')
parser.add_argument('--model_name', type=str, default='astmodel', help='Model to use')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--workers', type=int, default=4, help='Number of workers')
parser.add_argument('--targetLength', type=int, default=1024, help='Target length of audio')
parser.add_argument('--runs', type=int, default=3, help='Number of runs')

args = parser.parse_args()

if __name__ == '__main__':
    for run in range(args.runs):
        set_seed(run)
        args.run_name = f'run{run}'
        trainer = trainer(vars(args))
        trainer.run()