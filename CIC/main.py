# main.py

import argparse
import subprocess

def run_train(args):
    subprocess.run(["python", "train.py", 
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed)])  

def run_test(args):
    subprocess.run(["python", "test.py", 
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed)])  
    
def run_train_loss(args):
    subprocess.run(["python", "train_loss.py",
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed)])
def run_demo(args):
    subprocess.run(["python", "demo.py", "--img_path", args.i, "--use_gpu"])
def main():
    """ Arguments for train, train_loss and test"""
    parser = argparse.ArgumentParser(description='Main script to run different functionalities.')
    parser.add_argument('--input-path', type=str, help='Path to folder of training images', default='imgs/train/mycoco_train2017/')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, help='batch size per GPU', default=16)  # Adjusted batch size
    parser.add_argument('--epochs', type=int, help='number of epochs', default=2)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


    """ Arguments for demo"""
    
    parser.add_argument('--mode', type=str, help='Functionality to run (train, test, etc.)', required=True)
    parser.add_argument('--i', '--img_path', type=str, default='images/mycoco_val2017/000000002532.jpg')
    
    
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'test':
        run_test(args)
    elif args.mode == 'train_loss':
        run_train_loss(args)
    elif args.mode == 'demo':
        run_demo(args)
    else:
        print("Invalid type. Supported types: train, test.")

if __name__ == "__main__":
    main()