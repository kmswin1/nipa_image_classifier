from argparse import ArgumentParser


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument('-epochs', '--epochs', type=int, default=100000, help='the number of maximum training epoch')
    parser.add_argument('-model', '--model', type=str, default='resnext', help='lower-cased model name')
    parser.add_argument('-optimizer', '--optimizer', type=str, default='Adam', help='lower-cased optimizer string')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='initial learning rate of optimizer')
    parser.add_argument('-order', '--order', type=int, default=2, help='order of norm (1 or 2)')
    parser.add_argument('-num_processes', '--num_processes', type=int, default=4, help='the number of processes')
    parser.add_argument('-seed', '--seed', type=int, default=1234, help='random seed')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128, help='size of batch in mini-batch training')
    parser.add_argument('-save_step', '--save_step', type=int, default=10, help='step size of save check point of model')
    args = parser.parse_args()
    return args