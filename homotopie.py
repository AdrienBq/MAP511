import os
import time
import importlib
import argparse
import random

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData
from modules import VAE
from modules import GaussianLSTMEncoder, LSTMDecoder

from exp_utils import create_exp_dir
from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_mi, calc_au, sample_sentences, visualize_latent, reconstruct

clip_grad = 5.0
decay_epoch = 5
lr_decay = 0.5
max_decay = 5

logging = None



def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # decoding
    parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")


    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                         help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=0,
                        help="save checkpoint every epoch before this number")
    parser.add_argument("--save_latent", type=int, default=0)

    # new
    parser.add_argument("--fix_var", type=float, default=-1)
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--load_best_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")

    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    # set seeds
    # seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    # args.seed = seed_set[args.taskid]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    load_str = "_load" if args.load_path != "" else ""
    if args.fb == 0:
        fb_str = ""
    elif args.fb == 1:
        fb_str = "_fb"
    elif args.fb == 2:
        fb_str = "_fbdim"
    elif args.fb == 3:
        fb_str = "_fb3"

    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}{}/{}_warm{}_kls{:.1f}{}_tr{}".format(args.dataset,
            load_str, args.dataset, args.warm_up, args.kl_start, fb_str, args.target_kl)


    if len(args.load_path) <= 0 and args.eval:
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args

def main(filename1, filename2, args):
    file1  = open(filename1, 'r')
    lines1 = file1.readlines()
    file1.close()

    file2  = open(filename2, 'r')
    lines2 = file2.readlines()
    file2.close()

    if len(lines1)!=len(lines2):
        raise ValueError("the two texts don't have the same number of line")

    idx = random.randint(0,len(lines1)-1)

    line1 = lines1[idx]
    line2 = lines2[idx]

    new_file = open('homotopie_file.txt', 'w')
    new_file.write(line1)
    new_file.write(line2)
    new_file.close()

    global logging
    debug = (args.reconstruct_from != "" or args.eval == True) # don't make exp dir for reconstruction
    logging = create_exp_dir(args.exp_dir, scripts_to_save=None, debug=debug)

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    
    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device

    test_data = MonoTextData( 'homotopie_file.txt',label = args.label, vocab = vocab)
    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    if args.enc_type == 'lstm':
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path)
        #curr_state_dict = vae.state_dict()
        #curr_state_dict.update(loaded_state_dict)
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)
    
    z1, _ = vae.encode(test_data_batch[0])
    z2, _ = vae.encode(test_data_batch[1])

    z1 = z1.squeeze(0)
    z2 = z2.squeeze(0)


    file = open('homotopie_ex.txt', 'w')
    file.write(line1)
    

    for lamb in np.linspace(0,1,100):
        z = z1
        for i in range(z.size()[0]):
            z[i] = (1-lamb)*z1[i] + lamb*z2[i]
        
        line = vae.decode(z,"sample")
        for word in line[0][0:-1]:
            file.write(word+' ')
        file.write('\n')
        

    file.write(line2)

    file.close()

if __name__ == '__main__':
    args = init_config()
    filename1 = 'Tatoeba.fr.txt'
    filename2 = 'Tatoeba.en.txt'
    main(filename1, filename2,args)