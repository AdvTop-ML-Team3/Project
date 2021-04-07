import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import data_handler
import specifications
from model import GGNN


def iteration(args, net, output, criterion, data_loaded):
    if args.model == 'ggnn':
        adj_matrix, annotation, target = data_loaded
        padding = torch.zeros(len(annotation), args.n_node_ids,
                              args.hidden_layer_dim - specifications.ANNOT_DIMENSION[str(args.task_id)]).double()
        t = Variable(target).long()
        pred = net(Variable(torch.cat((annotation, padding), 2)), Variable(annotation), Variable(adj_matrix))
        if args.task_id == 19:
            # For task 19, an output prediction is made for each step
            pred = pred.view(-1, pred.shape[-1])
            t = t.view(-1)
    else:
        x, y = data_loaded
        t = Variable(y).long()
        pred = net(x)[1] if args.model == 'rnn' else net(x)[1][1]
        pred = pred.view(-1, args.hidden_layer_dim)
    loss = criterion(pred, t)
    output['loss'].append(loss.item())
    output['accuracy'].append((pred.max(1)[1] == t).float().mean().item())
    return loss


def train(args):
    babi_split, babi_loader = data_handler.get_data_loader(args, mode='train')
    net_dict = {
        'ggnn': lambda: GGNN(args).double(),
        'lstm': lambda: nn.LSTM(args.hidden_layer_dim, args.hidden_layer_dim, batch_first=True),
        'rnn': lambda: nn.RNN(args.hidden_layer_dim, args.hidden_layer_dim, batch_first=True),
    }
    net = net_dict[args.model]().train()
    output = {'net': net}
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    print_msg('-------------------- Calculating for Dataset:', args.dataset_id, '---------------------------')
    for i in range(args.epochs):
        output['loss'], output['accuracy'] = [], []
        for data_loaded in babi_loader:
            loss = iteration(args, net, output, nn.CrossEntropyLoss(), data_loaded)
            net.zero_grad()
            loss.backward()
            optimizer.step()
        print_msg('Train Epoch:', i, 'Loss: {:.3f} '.format(np.mean(output['loss'])),
                  'Accuracy: {:.3f} '.format(np.mean(output['accuracy'])))
    return output


def predict(args, trained_model):
    babi_split, babi_loader = data_handler.get_data_loader(args, mode='test')
    net = trained_model.eval()
    output = {'loss': [], 'accuracy': []}
    for data_loaded in babi_loader:
        iteration(args, net, output, nn.CrossEntropyLoss(), data_loaded)
    return output

def print_msg(*args):
    if specifications.VERBOSE:
        print(*args)
