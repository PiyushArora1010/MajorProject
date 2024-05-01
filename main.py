import os
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train import trainer

parser = argparse.ArgumentParser(description='Audio Classification')

parser.add_argument('--dataset', type=str, default='urbansound', help='Dataset to use')
parser.add_argument('--model_name', type=str, default='astmodel', help='Model to use')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--workers', type=int, default=4, help='Number of workers')
parser.add_argument('--targetLength', type=int, default=1024, help='Target length of audio')
parser.add_argument('--run_name', type=str, default='model', help='Name of the model')

args = parser.parse_args()

if __name__ == '__main__':
    trainer = trainer(vars(args))
    trainer.run()