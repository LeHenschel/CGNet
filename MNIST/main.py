import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import argparse
import os
import time
import datetime
import MNIST.model as MNIST_model
import MNIST.dataloader as MNIST_dld
from torch.utils.data import DataLoader
import sys
import tqdm
import functools
import _settings
import ipdb
import csv
import glob

print("Cuda available?", torch.cuda.is_available())

TRAIN = 'train'
VALID = 'val'
TEST = 'test'

import MNIST.MNIST_data_process as MNIST_datautils
# from CGNet.SphericalCNN_fast import SphericalCNN_fast as SphericalCNN
# from CGNet.SphericalCNN_fast import SphericalResCNN_fast as SphericalResCNN
from CGNet.CGNet import SphericalCNN as SphericalCNN


def argument_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser(description='train the network')
    parser.add_argument('--nlayers', type=int, default=5,
                        help='number of layers if not ResNet; number of layer for each lmax in ResNet')
    parser.add_argument('--tau_type', type=int, default=3,
                        help='how to set tau for each layer. Choose from 1: all tau==tau_man; 2: ceil(tau_man/(2l+1)); 3: ceil(tau_man/sqrt(2l+1))')
    parser.add_argument('--tau_man', type=int, default=12, help='see description of tau_type')
    parser.add_argument('--logPath', default=None, help='log path')
    parser.add_argument('--norm', type=int, default=1,
                        help='normalization strategy: \n' + \
                             '0: no normlization \n' + \
                             '1: batch-normalization in each layer by scaling each fragment down by a moving average of standard deviation\n')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help="Directory to store checkpoints. Will be automatically determined if not set based on selected options")
    parser.add_argument('--base_dir', type=str, default=None,
                        help="Base Directory to store checkpoints in. Will be automatically determined if not set based on selected options")
    parser.add_argument('--resume', action="store_true", default=False,
                        help='Flag, if training should be resumed for the given checkpoint.')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--num-epoch', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight_decay rate (default: 1e-5)')
    parser.add_argument('--lmax', type=int, default=11,
                        help='lmax')
    parser.add_argument('--skip', type=int, default=1,
                        help='0: no skipping; 1: concat output of all layers (l=0 for all layers); 2: ResNet')

    parser.add_argument('--unrot-test', action="store_true", default=False,
                        help='if True, measure on unrotated test set')

    parser.add_argument('--rotate-train', action="store_true", default=False, help='train on rotated training set')

    parser.add_argument('--dropout', type=str, default="0.5",
                        help='If set, dropout in the fully connected layers with this probability. ')
    parser.add_argument('--nfc', type=int, default=1, help='number of fully connected layers')
    parser.add_argument('--data_dir', type=str, default=_settings.MNIST_PATH,
                        help="Directory with training/testing files.")
    parser.add_argument('--csv_file', type=str, default=None, help="Path where to save run information to csv")
    parser.add_argument('--mst_weight', type=str, default="cost", choices=["cost", "random", "none", "sum"],
                        help="Define way to compute edge weights for MST. "
                             "Options: cost = computational costs, sum = sum of ls (l+l1+l2), "
                             "         random = random weights (randomly sampled in interval [1, 10],"
                             "         none = unweighted (same weight of 1 for all edges)")
    parser.add_argument('--mst', action="store_true", default=False, help="Reduce CG fragments via MST. Default: False")
    parser.add_argument('--py', action="store_true", default=False,
                        help="Use the python version of SphericalCNN (not CUDA optimized)")
    sel_option = parser.parse_args()
    return sel_option


# python main.py --nlayers 5 --tau_type 2 --batch-size 5 --num-epoch 2 --lmax 3 --batch-norm


def eval_data_and_log(net, data, label, criterion, eval_err=None, logger_name="log_train", extra_info=""):
    def eval_pred_error(output, target):
        n = target.size(0)
        # ipdb.set_trace()
        predict = torch.max(output.data, 1)[1].squeeze()
        t = torch.eq(predict, target.data).float()
        return 1. - t.sum() / float(n)

    if eval_err is None:
        eval_err = eval_pred_error
    logger = logging.getLogger(logger_name)
    output = net(data)
    loss = criterion(output, label)
    # print(loss)
    err = eval_err(output, label)
    logger.info(" loss={}, error={}. time={}. ".format(loss.item(), err, datetime.datetime.now()) + extra_info)
    return loss, err


# small helper
def _update_hist(hist, cur):
    _get_float = lambda x: x if isinstance(x, float) else x.item()
    hist[0].append(_get_float(cur[0]))
    hist[1].append(_get_float(cur[1]))


def eval_data_in_batch_new(net, dataloader, criterion, extra_info="", logger_name="log_train"):
    logger = logging.getLogger(logger_name)
    logger.info("Training? {}".format(net.training))
    history = [[], []]

    net.eval()
    with torch.no_grad():
        st = 0
        for data, target, _ in tqdm.tqdm(dataloader, desc='Eval'):
            st += len(target)
            results = eval_data_and_log(net, data.cuda(), target.cuda(), criterion, logger_name="log_train",
                                        extra_info="{}/{}".format(st, len(dataloader.dataset)))
            _update_hist(history, results)
    overall_loss = np.mean(np.asarray(history[0]))
    overall_err = np.mean(np.asarray(history[1]))
    logger.info("average loss={}, error={}. ".format(overall_loss, overall_err) + extra_info)
    return overall_loss, overall_err


# from MPNN
def save_net(state, is_best, directory):
    import shutil
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'last_check_point.pth')
    best_net_file = os.path.join(directory, 'net_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_net_file)


def main(args, save_period=4000):
    # log_name = args.logPath.split("/")[-1].split(".")[0]
    logging.basicConfig(filename=args.logPath, level=logging.INFO)
    # data_train,label_train,data_valid,label_valid,data_test,label_test=prepare_datasets(args,N_TRAIN=N_TRAIN,N_TEST=N_TEST)
    train_dataset, val_dataset = [
        MNIST_dld.MNISTData(mode=s, lmax=args.lmax, rotate=args.rotate_train, data_dir=args.data_dir) for s in
        [TRAIN, VALID]]
    test_dataset = MNIST_dld.MNISTData(mode=TEST, lmax=args.lmax, rotate=not args.unrot_test, data_dir=args.data_dir)
    import utils.utils as utils;
    utils.set_all_seeds(7)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger = logging.getLogger("log_main")
    csv_save_dict = {"mst": args.mst, "mst_weight": args.mst_weight}
    # assert args.skip == 1 and args.norm == 1
    model = MNIST_model.MNIST_Net(args.lmax - 1, args.tau_type, args.tau_man,
                                  args.nlayers, skipconn=True, norm=True, cuda=True,
                                  dropout=args.dropout, nfc=args.nfc, sparse=args.mst,
                                  weight_type=args.mst_weight, py=args.py)
    MNIST_model.show_num_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # added weigth decay
    criterion = nn.NLLLoss()
    best_err = cur_err = 1.0
    args.start_epoch = 0
    args.st = 0
    if args.resume:
        logger.info("Resume Training run {}".format(args.ckpt_dir))
        last_net_file = os.path.join(args.ckpt_dir, 'last_check_point.pth')
        if os.path.isfile(last_net_file):
            logger.info("=> loading last model '{}'".format(last_net_file))
            checkpoint = torch.load(last_net_file)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_err']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            args.st = checkpoint['st']
            logger.info(
                "=> loaded last model '{}' (epoch {}, st={})".format(last_net_file, checkpoint['epoch'], args.st))
        else:
            logger.info("=> no stored model found at '{}'".format(last_net_file))

    train_history = [[], []]  # [0] is history of losses, [1] is errors
    valid_history = [[], []]
    start_time = datetime.datetime.now()
    for epoch in range(args.start_epoch, args.num_epoch):
        last_save_pt = 0
        st = 0
        for data, target, _ in tqdm.tqdm(train_loader, desc='Training Epoch=%d' % epoch, ncols=80):
            st += len(target)
            if st < args.st + len(target): continue
            ed = st + len(target)
            model.train()

            optimizer.zero_grad()
            results = eval_data_and_log(model, data.cuda(), target.cuda(), criterion, logger_name="log_train",
                                        extra_info="{}/{}".format(st, len(train_loader.dataset)))
            _update_hist(train_history, results)
            results[0].backward()
            optimizer.step()

            if (ed - last_save_pt >= save_period) and st > 0:
                logging.info("Saving at {}: best_err={}".format(datetime.datetime.now(), best_err))
                last_save_pt = ed
                save_net({'epoch': epoch, 'state_dict': model.state_dict(), 'best_err': min(cur_err, best_err),
                          'optimizer': optimizer.state_dict(), 'st': last_save_pt}, False, args.ckpt_dir)
        # eval on validation set and save
        model.eval()
        results = eval_data_in_batch_new(model, valid_loader, criterion, logger_name="log_valid")
        _update_hist(valid_history, results)
        cur_err = valid_history[1][-1]
        print("Curr Err = %f" % cur_err)
        save_net({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_err': min(cur_err, best_err),
                  'optimizer': optimizer.state_dict(), 'st': 0}, cur_err < best_err, args.ckpt_dir)
        if cur_err < best_err:
            best_err = cur_err
        elif cur_err > best_err * 2:
            logging.info(
                "epoch {}: validation error {} > best error {} a lot, quitting as it might be overfitting".format(epoch,
                                                                                                                  cur_err,
                                                                                                                  best_err))
            break
        logging.info("epoch {} done, Time{}".format(epoch, datetime.datetime.now()))
        args.st = 0
    end_time = datetime.datetime.now()
    csv_save_dict["train_time"] = (end_time - start_time).microseconds
    csv_save_dict["best_error"] = best_err

    # Testing phase --> Load best trained model
    logger.info("Resume Testing on best run in {}".format(args.ckpt_dir))
    best_net_file = os.path.join(args.ckpt_dir, 'net_best.pth')
    if os.path.isfile(best_net_file):
        logger.info("=> loading best model '{}'".format(best_net_file))
        checkpoint = torch.load(best_net_file)
        best_err = checkpoint['best_err']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded best model with valid_err={}".format(best_err))
    else:
        logger.info("=> no best model found at '{}'".format(best_net_file))

    model.eval()
    # test_error = eval_data_in_batch(model, data_test, label_test,criterion,logger_name="log_test",batch_size=args.batch_size)[1]
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_error = eval_data_in_batch_new(model, test_loader, criterion, logger_name="log_test")[1]
    logger.info("FINAL test error = {}".format(test_error))

    if args.csv_file:
        if os.path.isfile(args.csv_file):
            with open(args.csv_file, 'r+') as f:
                header = next(csv.reader(f))
                dict_writer = csv.DictWriter(f, header, -999)
                dict_writer.writerow(csv_save_dict)
        else:
            with open(args.csv_file, 'w') as f:
                writer = csv.DictWriter(f, csv_save_dict.keys())
                writer.writeheader()
                writer.writerow(csv_save_dict)

    return train_history, valid_history

# newest
if __name__ == "__main__":
    args = argument_parse()

    args.dropout = None if args.dropout == "None" else float(args.dropout)

    if args.skip == 2:
        train_name = "ResNet_epoch{}_lr{}_wd{}_layerstep{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}_{}_edgeW{}".format(
            args.num_epoch, args.lr, args.weight_decay, args.nlayers, args.tau_type, args.tau_man, args.lmax,
            args.batch_size, args.norm, args.nfc, "MST" if args.mst else "Full", args.mst_weight)
    else:
        train_name = "epoch{}_lr{}_wd{}_layer{}_tau{}-{}_lmax{}_batchsize{}_norm{}_nfc{}_{}_edgeW{}".format(args.num_epoch,
                                                                                                 args.lr,
                                                                                                 args.weight_decay,
                                                                                                 args.nlayers,
                                                                                                 args.tau_type,
                                                                                                 args.tau_man,
                                                                                                 args.lmax,
                                                                                                 args.batch_size,
                                                                                                 args.norm, args.nfc,
                                                                                                 "MST" if args.mst else "Full",
                                                                                                 args.mst_weight)
        train_name += "" if args.skip == 0 else "_connect-to-output"
    train_name += "_new"
    # train_name += "_relu" if args.relu else ""
    if args.rotate_train and (not args.unrot_test):
        train_name += "R-R"
    elif (not args.rotate_train) and (not args.unrot_test):
        train_name += "NR-R"
    else:
        assert ((not args.rotate_train) and args.unrot_test)
        train_name += "NR-NR"
    train_name += "" if args.dropout is None else "_dropout{}".format(args.dropout)

    if args.base_dir is None:
        CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    else:
        CUR_DIR = args.base_dir

    if args.logPath is None:
        args.logPath = CUR_DIR + "/temp_fast/logs/{}.log".format(train_name)
    if args.ckpt_dir is None:
        args.ckpt_dir = CUR_DIR + "/temp_fast/checkpoint/{}/".format(train_name)

    log_dir = os.path.dirname(args.logPath)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    else:
        # preserve the best net by adding a suffix to it
        element = len(glob.glob(os.path.join(args.ckpt_dir, "net_best*.pth")))
        try:
            os.rename(os.path.join(args.ckpt_dir, "net_best.pth"), os.path.join(args.ckpt_dir, "net_best_{}.pth".format(element)))
        except FileNotFoundError:
            print("Did not find previous run file, continue with training")

    main(args, save_period=20000)
