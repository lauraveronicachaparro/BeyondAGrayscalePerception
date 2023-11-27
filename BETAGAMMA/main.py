# main.py

import argparse
import subprocess

def run_train(args):
    subprocess.run(["python", "train.py", 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--momentum", str(args.momentum),
                    "--gamma", str(args.gamma),
                    "--log-interval", str(args.log_interval),
                    "--save", str(args.save)])  

def run_test(args):
    subprocess.run(["python", "test.py",
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--momentum", str(args.momentum),
                    "--gamma", str(args.gamma),
                    "--log-interval", str(args.log_interval),
                    "--save", str(args.save)])  
    
def run_train_loss(args):
    subprocess.run(["python", "train_loss.py",
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--momentum", str(args.momentum),
                    "--gamma", str(args.gamma),
                    "--log-interval", str(args.log_interval),
                    "--save", str(args.save)])
def run_demo(args):
    subprocess.run(["python", "demo.py", "--img_path", args.i,
                    "--input-path", args.input_path, 
                    "--batch-size", str(args.batch_size), 
                    "--epochs", str(args.epochs),
                    "--seed", str(args.seed),
                    "--lr", str(args.lr),
                    "--momentum", str(args.momentum),
                    "--gamma", str(args.gamma),
                    "--log-interval", str(args.log_interval),
                    "--save", str(args.save)])
def main():
    """ Arguments for train, train_loss and test"""
    parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=2, metavar='M',
                        help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--save', type=str, default='models/modeloTodo20.pt',
                        help='file on which to save model weights')


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