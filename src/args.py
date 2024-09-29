import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=384,
                        help='Number of hidden units.')
    parser.add_argument('--out', type=int, default=200,
                        help='Number of out.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="MHGCN-ChebNetII",
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--K', type=int, default="10",
                        help='伯恩斯坦多项式项数')
    parser.add_argument('--dprate', type=float, default="0.5",
                        help='伯恩斯坦多项式项数')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')


    parser.add_argument('--dset', type=str, default='imdb', choices=['acmv9', 'dblpv7', 'citationv1'])
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--output', type=str, default='ckps/source')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--output_adapt', type=str, default='ckps/adapt')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int, default=40, help="max adapt iterations")
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args