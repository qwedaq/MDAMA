import random
import time
import warnings
import argparse
import os.path as osp
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import tllib.vision.datasets as datasets
import copy
from tllib.vision.transforms import MultipleApply
from scipy.spatial.distance import cdist
import numpy as np
from torch.autograd import Variable
import higher
from torch.autograd import grad
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Multisource")

def obtain_label(loader, model, distance='cosine', epsilon=1e-5, threshold=0, log=False, num_step=1):
    start_test = True
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, _ = data[:2]
            inputs = images.to(device)
            feas, outputs = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(num_step):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    if log:
        return pred_label.astype('int')
    else:
        return pred_label.astype('int'), initc, initc[labelset]

def wt_crossentropy(input_,labels):
    input_s = F.softmax(input_,dim=1)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    confidence, pseudo_labels = F.softmax(input_.detach(), dim=1).max(dim=1)
    mask = (confidence < 0.9).float()
    return torch.mean(mask*weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))

def mixup_data(x_s1,x_s2,x_s3,x_t,ratio1, ratio2, ratio3):
    src = torch.cat((x_s1,x_s2,x_s3),dim=0)
    x_mix = (0.7*x_t)+(0.3*src)
    return x_mix 

def mixup_data_s(x_s1,x_s2,x_s3,ratio1, ratio2, ratio3):
    x_mix = (ratio1 * x_s1) +(ratio2 * x_s2) +(ratio3 * x_s3)
    return x_mix 

def gradientdiscrepancy(cls_loss1,cls_loss2,cls_loss3,cls_loss_mix,classifier):
    grad_sim=[]
    hess_sim =[]
    names=['head.linear_h.weight','head.linear_h.bias','head.linear2_h.weight','head.linear2_h.bias']

    for n, p in classifier.named_parameters():
        if n in names:
            source1_grad = grad([cls_loss1],[p],create_graph=True,only_inputs=True,allow_unused=False)[0]
            source2_grad = grad([cls_loss2],[p],create_graph=True,only_inputs=True,allow_unused=False)[0]
            source3_grad = grad([cls_loss3],[p],create_graph=True,only_inputs=True,allow_unused=False)[0]
            sourcemix_grad = grad([cls_loss_mix],[p],create_graph=True,only_inputs=True,allow_unused=False)[0]
            
            grad_avg = (source1_grad + source2_grad + source3_grad+ sourcemix_grad)/4.0

            if len(p.shape) > 1:
                _cossim1 = F.cosine_similarity(grad_avg, source1_grad, dim=1).mean()
                _cossim2 = F.cosine_similarity(grad_avg, source2_grad, dim=1).mean()
                _cossim3 = F.cosine_similarity(grad_avg, source3_grad, dim=1).mean()
                _cossim_mix = F.cosine_similarity(grad_avg, sourcemix_grad, dim=1).mean()
                _cossim = _cossim1 +_cossim2 +_cossim3+_cossim_mix
            else:
                _cossim1 = F.cosine_similarity(grad_avg, source1_grad, dim=0)
                _cossim2 = F.cosine_similarity(grad_avg, source2_grad, dim=0)
                _cossim3 = F.cosine_similarity(grad_avg, source3_grad, dim=0)
                _cossim_mix = F.cosine_similarity(grad_avg, sourcemix_grad, dim=0)
                _cossim = _cossim1 +_cossim2 +_cossim3 + _cossim_mix
            grad_sim.append(_cossim)
    
    grad_cos = torch.stack(grad_sim)
    grad_loss = (1.0 - grad_cos).sum()

    return grad_loss/10

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    meta = False
    gen_meta = False

    # Data loading code
    train_source_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    weak_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                             random_horizontal_flip=not args.no_hflip,
                                             random_color_jitter=False, resize_size=args.resize_size,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                               random_horizontal_flip=not args.no_hflip,
                                               random_color_jitter=False, resize_size=args.resize_size,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std,
                                               auto_augment=args.auto_augment)
    train_target_transform = MultipleApply([weak_augment, strong_augment])

    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_source_transform: ", train_source_transform)
    print("train_target_transform",train_target_transform)
    print("val_transform: ", val_transform)

    train_source1_dataset,train_source2_dataset,train_source3_dataset,train_source4_dataset,train_source5_dataset,_, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_source_transform, val_transform, multi= True,train_target_transform=train_target_transform)
    
    train_source1_loader = DataLoader(train_source1_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_source2_loader = DataLoader(train_source2_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_source3_loader = DataLoader(train_source3_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size_tgt,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_tgt, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_tgt, shuffle=False, num_workers=args.workers)

    train_source1_iter = ForeverDataIterator(train_source1_loader)
    train_source2_iter = ForeverDataIterator(train_source2_loader)
    train_source3_iter = ForeverDataIterator(train_source3_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    sum_ = train_source1_dataset.__len__() + train_source2_dataset.__len__() +train_source3_dataset.__len__()
    ratio1 = train_source1_dataset.__len__()/sum_
    ratio2 = train_source2_dataset.__len__()/sum_
    ratio3 = train_source3_dataset.__len__()/sum_

    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim, pool_layer=pool_layer).to(device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)
    print("POOL LAYER: "+str(pool_layer) )
    # define optimizer and lr_scheduler
    # The learning rate of the classiﬁers are set 10 times to that of the feature extractor by default.
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source1_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    #Training loop
    best_acc1 = 0.
    min_epoch = 1
    meta_true = False
    mem_label=None
    print("Source Domain: {}".format(args.source))
    print("Target Domain: {}".format(args.target))

    for epoch in range(args.epochs):
        print("warm up__: "+str(meta_true))

        if epoch < min_epoch:
            # train for one epoch
            train(train_source1_iter,train_source2_iter,train_source3_iter, train_target_iter, None, classifier, mdd, optimizer, None, meta_true,epoch, val_loader, num_classes,ratio1,ratio2,ratio3,
              lr_scheduler, epoch, args)
            meta_true = True
        else:
            train(train_source1_iter,train_source2_iter,train_source3_iter, train_target_iter, mem_label, classifier, mdd, optimizer, mem_label, meta_true,epoch, val_loader,num_classes,ratio1,ratio2,ratio3,
              lr_scheduler, epoch, args)
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        mem_label,_,_ = obtain_label(val_loader,classifier)
        mem_label = Variable(torch.from_numpy(mem_label)).cuda()
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

        print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source1_iter: ForeverDataIterator, train_source2_iter: ForeverDataIterator, train_source3_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, meta_iter: ForeverDataIterator,
          classifier: ImageClassifier, mdd: MarginDisparityDiscrepancy, optimizer: SGD, mem_label,meta_true, curr_ep, val_loader,class_num,ratio1,ratio2,ratio3,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
    # switch to train mode
    classifier.train()
    mdd.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        x_s1, labels_s1 = next(train_source1_iter)[:2]
        x_s2, labels_s2 = next(train_source2_iter)[:2]
        x_s3, labels_s3 = next(train_source3_iter)[:2]
        
        (x_t,x_t_strong),_,_,tgt_idx = next(train_target_iter)[:4]
       
        if meta_true:
            psuedo_labels = mem_label[tgt_idx]

        x_s1 = x_s1.to(device)
        x_s2 = x_s2.to(device)
        x_s3 = x_s3.to(device)
        
        x_t = x_t.to(device)
        x_t_strong=x_t_strong.to(device)
        labels_s1 = labels_s1.to(device)
        labels_s2 = labels_s2.to(device)
        labels_s3 = labels_s3.to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        x_mix = mixup_data(x_s1,x_s2,x_s3,x_t,ratio1,ratio2,ratio3)
        x_mix_s = mixup_data_s(x_s1,x_s2,x_s3,ratio1,ratio2,ratio3)
        # compute output
        x = torch.cat((x_s1,x_s2,x_s3,x_mix_s,x_mix, x_t,x_t_strong), dim=0)
        outputs, outputs_adv = classifier(x)
        src,tgt = outputs[:40],outputs[40:]
        src_adv,tgt_adv = outputs_adv[:40],outputs_adv[40:]
        y_s1,y_s2,y_s3,y_mix_s = src.chunk(4, dim=0)
        y_s1_adv,y_s2_adv,y_s3_adv,_ = src_adv.chunk(4, dim=0)
        y_mix, y_t,y_t_strong = tgt.chunk(3,dim=0)
        y_mix_adv, y_t_adv,_=tgt_adv.chunk(3,dim=0)

        y_s = torch.cat((y_s1,y_s2,y_s3),dim=0)
        y_s_adv = torch.cat((y_s1_adv,y_s2_adv,y_s3_adv),dim=0)
        x_s = torch.cat((x_s1,x_s2,x_s3), dim=0)
        labels_s = torch.cat((labels_s1,labels_s2,labels_s3),dim=0)
        # compute cross entropy loss on source domain
        cls_loss1 = F.cross_entropy(y_s1,labels_s1)
        cls_loss2 = F.cross_entropy(y_s2,labels_s2)
        cls_loss3 = F.cross_entropy(y_s3,labels_s3)
        mix_loss = (ratio1 * F.cross_entropy(y_mix_s,labels_s1)) + (ratio2 * F.cross_entropy(y_mix_s,labels_s2)) + (ratio3 * F.cross_entropy(y_mix_s,labels_s3))
        
        cls_loss_ = (cls_loss1 + cls_loss2 + cls_loss3)/3.0
        cls_loss_ += mix_loss*0.5
        if meta_true:
            self_training_loss, mask, pseudo_labels_ = self_training_criterion(y_t_strong, y_t)
            cls_loss_ +=(self_training_loss)
            cls_loss_mix = 0.7*((mask*(F.cross_entropy(y_mix,pseudo_labels_,reduction='none'))).mean()) + 0.3*((mask*(F.cross_entropy(y_mix,labels_s,reduction='none'))).mean())
            ent_loss = wt_crossentropy(y_t,psuedo_labels)
            cls_loss_ += (ent_loss)*0.5 
            cls_loss_ += (cls_loss_mix)
            cls_loss_ += (gradientdiscrepancy(cls_loss1,cls_loss2,cls_loss3,cls_loss_mix,classifier))
            
        transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
        
        loss = cls_loss_ + transfer_loss * args.trade_off
        classifier.step()

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('-m','--meta',help='meta domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N',
                        help='mini-batch size (default: 10)')
    
    parser.add_argument('-bt', '--batch-size-tgt', default=30, type=int,
                        metavar='Nt',
                        help='mini-batch size (default: 30)')
    
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='confidence threshold')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mdd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)