#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torchvision
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn.functional as F
from Imagefolder_modified import Imagefolder_modified
from Imagefolder_meta import Imagefolder_meta
from resnet_modified import SNet,LNet, ResNet18_meta, ResNet34_meta
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True


import time

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

class Manager(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
            path    [dict]  path of the dataset and model
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        # model path
        os.popen('mkdir -p ' + self._path)
        #training set
        self._data_base = options['data_base']
        #validation set
        self._validation_base = options['validation_base']
        self._class = options['n_classes']
        self._ts = options['ts']
        self._drop_rate = options['drop_rate']
        self._relabel_rate = options['relabel_rate']
        self._stop_snet = options['stop_snet']
        self._meta_number = options['meta_number']
        self._plus = options['plus']
        print('Basic information: ','data:',self._data_base,'  lr:', self._options['base_lr'],'  w_decay:', self._options['weight_decay'])
        print('Parameter information: ','drop_rate:',self._drop_rate, 'relabel_rate:',self._relabel_rate,'  ts:',self._ts)
        print('------------------------------------------------------------------------------')
        # Network
        print('network:', options['net'])
        if options['net'] == 'resnet18':
            self._NET = ResNet18_meta
        elif options['net'] == 'resnet34':
            self._NET = ResNet34_meta
        else:
            raise AssertionError('Not implemented yet')

        snet = SNet(512,256, 1)
        lnet = LNet(512, 200)
        net = self._NET(n_classes=options['n_classes'], pretrained=True)

        self._net = net.cuda()
        self._snet = snet.cuda()
        self._lnet = lnet.cuda()

        lr_f = self._options['lr_f']
        lr_label = self._options['base_lr']
        print('lr_b:',lr_f)

        self._optimizer_a = torch.optim.SGD(self._net.params(), lr=self._options['base_lr'], momentum=0.9, weight_decay=self._options['weight_decay'])
        self._optimizer_l = torch.optim.SGD(self._lnet.parameters(), lr=lr_label, momentum=0.9, weight_decay=self._options['weight_decay'])
        self._optimizer_s = torch.optim.SGD(self._snet.parameters(), lr=lr_f, momentum=0.9, weight_decay=self._options['weight_decay'])

        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_a, T_max=self._options['epochs'])
        self._scheduler_l = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_l,  T_max=self._options['epochs'])
        print('lr_scheduler: CosineAnnealingLR')

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Load data; if chashed = True, images will be loaded into RAM
        self._test_data = Imagefolder_modified(os.path.join(self._data_base, 'val'), transform=test_transform, cached=False)
        self._train_validation_data= Imagefolder_meta(os.path.join(self._data_base, 'train'), os.path.join(self._validation_base), transform=train_transform, number = self._meta_number,cached=False)

        # if Plus, validation set will be utilized in training classier network
        if self._plus:
            self._validation_data= Imagefolder_modified(os.path.join(self._validation_base), transform=train_transform, number = self._meta_number, cached=False)
            self._validation_loader = DataLoader(self._validation_data, batch_size=self._options['batch_size'],
                                                 shuffle=True, num_workers=4, pin_memory=True)

        print('number of classes in trainset is : {}'.format(len(self._train_validation_data.classes)))
        print('number of classes in testset is : {}'.format(len(self._test_data.classes)))
        assert len(self._train_validation_data.classes) == options['n_classes'] and len(self._test_data.classes) == options['n_classes'], 'number of classes is wrong'

        self._train_validation_loader = DataLoader(self._train_validation_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(self._test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)
        self._pin = torch.zeros(len(self._train_validation_data))
        self._most_prob_label = -1 * torch.ones(len(self._train_validation_data), dtype=torch.long)

    def _label_smoothing_cross_entropy(self,logit, label, epsilon=0.1, reduction='mean'):
        N = label.size(0)
        C = logit.size(1)
        smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
        smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1 - epsilon)

        if logit.is_cuda:
            smoothed_label = smoothed_label.cuda()

        log_logit = F.log_softmax(logit, dim=1)
        losses = -torch.sum(log_logit * smoothed_label, dim=1)  # (N)
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.sum(losses) / N
        elif reduction == 'sum':
            return torch.sum(losses)
        else:
            raise AssertionError('reduction has to be none, mean or sum')

    def _selection_loss_minibatch(self, logits, labels):
        loss_all = F.cross_entropy(logits, labels, reduction='none')
        index_sorted = torch.argsort(loss_all.detach(), descending=False)
        num_remember = int((1 - self._drop_rate) * labels.size(0))
        index_clean = index_sorted[:num_remember]
        logits_final = logits[index_clean]
        labels_final = labels[index_clean]

        loss = self._label_smoothing_cross_entropy(logits_final, labels_final, reduction = 'mean')
        return loss

    def _relabel_loss_minibatch(self, logits, labels, ids):
        loss_all = F.cross_entropy(logits, labels, reduction='none')

        index_sorted = torch.argsort(loss_all.detach(), descending=False)
        num_remember = int((1-self._drop_rate) * labels.size(0))
        index_clean = index_sorted[:num_remember]
        index_noise = index_sorted[num_remember:]

        weight_feature = self._pin[ids[index_noise]]
        noise_sorted = torch.argsort(weight_feature.detach(), descending=True)
        num_relabel = int(self._relabel_rate * labels.size(0))
        index_relabel = index_noise[noise_sorted[:num_relabel]]

        labels_clean = labels[index_clean]
        labels_relabel = self._most_prob_label[ids[index_relabel]].cuda()

        logits_final = logits[torch.cat((index_clean, index_relabel), 0)]
        labels_final = torch.cat((labels_clean, labels_relabel), 0)

        loss = self._label_smoothing_cross_entropy(logits_final, labels_final, reduction='mean')
        return loss, index_relabel, labels_relabel

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        print('Epoch\tTrain Loss\tTrain Accuracy\tVal Accuracy\tTest Accuracy\tLabel Accuracy\tEpoch Runtime')
        for t in range(self._options['epochs']):
            epoch_start = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            num_val = 0
            num_label=0

            for data in self._train_validation_loader:
                self._net.train(True)
                X, y, id, path, x_validation, y_validation = data

                # Data
                X = X.cuda()
                y = y.cuda()

                x_validation = x_validation.cuda()
                y_validation = y_validation.cuda()

                # update snet, early stop
                if t < self._stop_snet  and self._relabel_rate > 0:
                    meta_net = self._NET(n_classes=options['n_classes'], pretrained=False).cuda()
                    meta_net.load_state_dict(self._net.state_dict())

                    y_f_hat, feature = meta_net(X)
                    cost = F.cross_entropy(y_f_hat, y, reduce=False)
                    cost_v = torch.reshape(cost, (len(cost), 1))

                    v_lambda = self._snet(feature.detach())
                    norm_c = torch.sum(v_lambda)
                    # normalized
                    v_lambda_norm = v_lambda / norm_c

                    l_f_meta = torch.sum(cost_v * v_lambda_norm)

                    meta_net.zero_grad()
                    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
                    meta_lr = self._optimizer_a.state_dict()['param_groups'][0]['lr']
                    meta_net.update_params(lr_inner=meta_lr, source_params=grads)  # Eq. 3
                    del grads

                    y_g_hat,_ = meta_net(x_validation)
                    l_g_meta = F.cross_entropy(y_g_hat, y_validation)
                    _, prediction = torch.max(y_g_hat.data, 1)
                    num_val += torch.sum(prediction == y_validation.data).item()

                    self._optimizer_s.zero_grad()
                    l_g_meta.backward()  # Eq. 4
                    self._optimizer_s.step()

                # update lnet
                with torch.no_grad():
                    _, feature = self._net(x_validation)
                outputlabels = self._lnet(feature)

                loss_l = self._label_smoothing_cross_entropy(outputlabels, y_validation, reduction='mean')

                _ , prediction_labelnet = torch.max(outputlabels, 1)
                num_label += torch.sum(prediction_labelnet == y_validation.data).item()

                self._optimizer_l.zero_grad()
                loss_l.backward()
                self._optimizer_l.step()

                # Forward pass, update h
                y_f, feature = self._net(X)

                # update p_in
                if t < self._stop_snet  and self._relabel_rate > 0:
                    with torch.no_grad():
                        w_f = self._snet(feature)
                        self._pin[id] = w_f[:, 0].cpu().detach()

                # update labels
                with torch.no_grad():
                    outputlabels = self._lnet(feature)
                _, prediction_labelnet = torch.max(outputlabels, 1)
                self._most_prob_label[id]=prediction_labelnet.cpu().detach()

                # relabel in-distribution noisy images and discard out-of-distribution ones
                if t >= self._ts and self._relabel_rate > 0:
                    loss, index_relabel, re_labels = self._relabel_loss_minibatch(y_f,y,id)
                # relabel_rate == 0, only discard
                elif t >= self._ts:
                    loss = self._selection_loss_minibatch(y_f,y)
                # initial epochs, using all training samples
                else:
                    loss = self._label_smoothing_cross_entropy(y_f,y, reduction='mean')

                _ , prediction = torch.max(y_f, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()

                # Backward
                self._optimizer_a.zero_grad()
                loss.backward()
                self._optimizer_a.step()
                epoch_loss.append(loss.item())

            if self._plus:
                for data in self._validation_loader:
                    self._net.train(True)
                    X, y, _, _ = data
                    # Data
                    X = X.cuda()
                    y = y.cuda()
                    y_f, _ = self._net(X)

                    loss = self._label_smoothing_cross_entropy(y_f, y, reduction='mean')

                    _, prediction = torch.max(y_f, 1)
                    num_total += y.size(0)
                    num_correct += torch.sum(prediction == y.data).item()

                    # Backward
                    self._optimizer_a.zero_grad()
                    loss.backward()
                    self._optimizer_a.step()

            # Record the test accuracy of each epoch
            test_accuracy = self.test(self._test_loader)
            train_accuracy = 100 * num_correct / num_total
            val_accuracy = 100 * num_val / num_total
            label_accuracy = 100 * num_label / num_total

            self._scheduler.step()
            self._scheduler_l.step()

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, options['net'] + '.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, val_accuracy, test_accuracy, label_accuracy,
                                                            epoch_end - epoch_start))

        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y,_,_ in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                # Prediction
                score,_ = self._net(X)
                _, prediction = torch.max(score, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--data_base', dest='data_base', type=str)
    parser.add_argument('--validation_base', dest='validation_base', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--lr_f', dest='lr_f', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=60)
    parser.add_argument('--drop_rate', type=float, default=0.35)
    parser.add_argument('--relabel_rate', type=float, default=0.1)
    parser.add_argument('--stop_snet', type=int, default=5)
    parser.add_argument('--ts', type=int, default=5)
    parser.add_argument('--meta_number', dest='meta_number', type=int, default=10)
    parser.add_argument('--plus', action='store_true', help='Turns on training on validation set', default=False)

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory ' + model + ' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    options = {
        'base_lr': args.lr,
        'lr_f': args.lr_f,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.w_decay,
        'path': path,
        'data_base': args.data_base,
        'validation_base': args.validation_base,
        'net': args.net,
        'n_classes': args.n_classes,
        'drop_rate': args.drop_rate,
        'relabel_rate':args.relabel_rate,
        'stop_snet':args.stop_snet,
        'ts': args.ts,
        'meta_number': args.meta_number,
        'plus': args.plus
    }
    manager = Manager(options)
    manager.train()