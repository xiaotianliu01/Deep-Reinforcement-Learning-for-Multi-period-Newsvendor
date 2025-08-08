import argparse


def get_config():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--exp_name", type=str, default='walmart_dqn')
    parser.add_argument("--log_dir", type=str, default='./results/walmart_dqn.txt')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma_optimizer", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_num", type=int, default=6)
    parser.add_argument("--K", type=int, default=300)

    parser.add_argument("--y_dim", type=int, default=40)
    parser.add_argument("--Y_max", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--critical_ratio", type=float, default=.5)
    
    return parser
