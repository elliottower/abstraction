import argparse
import os
import shutil
import random
import shutil
import time
import warnings
from enum import Enum
import numpy as np
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--sample_percent', type=float, default=0.2, help='percentage of neurons to sample from each layer for abstraction calculation')
parser.add_argument('--output_name', type=str, default="", help="Custom name for output folder and Q folder, allows for saving of cached values without overwriting every time")
parser.add_argument('--theta', type=float, default=0.75, help="Theta value for Q matrix: correlation value which neurons in the next layer must have towards the output in order to be kept nonzero")
parser.add_argument('--theta-list', nargs='+', type=float, help="Allows for passing in variable length list of theta values, starting from the last layer and going backwards")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model = models.__dict__[args.arch](pretrained=args.pretrained)
    
    # Create feature extractor from model (for obtaining intermediate representations)
    train_nodes, eval_nodes = get_graph_node_names(model)
    feature_extractor = create_feature_extractor(model, train_return_nodes=train_nodes, eval_return_nodes=eval_nodes)
    
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        feature_extractor.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            feature_extractor.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()
            
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(True),
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, val_dataset, model, feature_extractor, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, val_dataset, model, feature_extractor, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, val_dataset, model, feature_extractor, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        
        # Clear output folder 
        q_dir = os.getcwd() + "/Q" + args.output_name + "/"
        if os.path.isdir(q_dir):
            shutil.rmtree(q_dir)
        os.makedirs(q_dir, exist_ok=True)
        
        output_dir = os.getcwd() + "/output" + args.output_name + "/"
        os.makedirs(output_dir, exist_ok=True)
                
        # TEMP: only use the first ten classes
        N = 10
        
        # Dry run to get shapes of activations & number of layers
        X = torch.randn(1, 3, 224, 224).to(device)
        out = feature_extractor(X)
        
        num_layers = len(out.keys())
        layer_dims = [o.flatten().size(0) for o in out.values()]
        
        Q = [] # List of lists with dimensions [ num_classes , num_layers ], holding tensors of dimensions  [ # neurons in layer K , # neurons in layer K+1 ]
        Q_indices = []
        targets = []
        pred_activations = []  #  List of lists with dimensions [ num_classes, num_layers ], holding tensors of dimensions [ # subsampled neurons, 16 ]
        
        # Initialize indices to None and then re define in loop with data
        indices =  None

        # Set theta value for each layer (optional arg to specify different values through theta_list, starting at end)
        theta = torch.tensor(args.theta).repeat(num_layers)
        if args.theta_list is not None:
            for idx, elem in enumerate(args.theta_list):
                print("Element: ", elem)
                theta[-idx] = elem
        print("Theta: ", theta)

        # Sample one class at a time (1000 classes, 50 images for validation, 150 for test set)
        for t in range(N):
            class_sampler = torch.utils.data.SubsetRandomSampler(torch.arange(50*t, 50*(t+1)))
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=class_sampler)
            
            for i, (images, target) in enumerate(val_loader):                
                if args.gpu is not None: 
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                
                 # compute output
                output = model(images)
                print("Output shape: ", output.shape)
                
                torch.save(output.cpu(), f"output{args.output_name}/class{t}_predictions.pt")
                loss = criterion(output, target)

                # Extract intermediate outputs from batch
                out = feature_extractor(images)
                
                print("Target: ", target) # Debugging to confirm it's the right target
                targets.append(target)

                # List of lists for Q and pred_activations which we will fill in iteratively
                Q.append([])
                Q_indices.append([])
                pred_activations.append([]) 

                # Input activation will be [ batch_size, 3, 224, 224 ] -> [ batch_size, 3 * 224 * 224 ]
                # Define activations and subsampled indices for the entire data batch, use global indices 
                activations = [ out[key].flatten(start_dim=1) for key in out.keys() ]

                # Choose indices to sample (sample neurons for each class)
                # Important: only define it once, as we want same neurons for each class/sample
                
                if indices is None:
                    indices = [ torch.tensor(random.sample(range(activation.size(1)), int(25000 * args.sample_percent))) for activation in activations[:-3] ]
                    indices.append(torch.arange(activations[-3].size(1))) # avgpool layer (dim 512) 
                    indices.append(torch.arange(activations[-2].size(1))) # flatten layer (dim 512)
                    indices.append(torch.arange(activations[-1].size(1))) # fc layer (dim 1000)

                print("Length of indices: ", len(indices))
                print("Length of activations: ", len(activations))
                
                # --- Calculate Q values ---
                
                print(f"Layer {num_layers -1}")

                # Q matrix for output layer (Defined to be one when i = j = t, zero otherwise)
                Q_output = torch.zeros((layer_dims[-1], 1))
                Q_output[t] = 1 
                Q_output_indices = torch.LongTensor(t) 
                
                print(f"Dimensions of output neurons {layer_dims[-1]}")
                print(f"Dimensions of output Q matrix: {Q_output.shape}")
                print(f"Nonzero Q values: ", Q_output.count_nonzero())
                
                torch.save(Q_output, "Q" + args.output_name + f"/class{t}layer{num_layers - 1}.pt") 
                torch.save(Q_output_indices, "Q" + args.output_name + f"/class{t}layer{num_layers - 1}_indices.pt") 
                
                # Insert to front, start at the end and build in reverse order
                Q[t].insert(0, Q_output.to(device)) 
                Q_indices[t].insert(0, Q_output_indices.to(device))
                
                # Keep full activations for output layer
                act_current = activations[-1]
                pred_activations[t].insert(0, act_current[34:].to(device)) 
                        
                # Start at second to last layer, count down to zero
                for layer in reversed(range(num_layers - 1)):
                    print(f"Layer {layer}")
                    
                    # Take sample from current layer activations (first 34 images)
                    act_current = activations[layer][:, indices[layer]]
                    act_sample = act_current[:34].T
                    
                    # Put aside remaining 16 images for prediction
                    pred_activations[t].insert(0, act_current[34:].cpu()) 

                    # Take sample from next layer activations (first 34 images), don't need to save prediction activations 
                    act_next_sample = activations[layer+1][:, indices[layer + 1]][:34].T
                    
#                     print(f"Dimensions of sampled neurons: {act_sample.shape},  {act_next_sample.shape}")

                    # For every neuron in layer K+1, check if it has as a value in the next layer's Q matrix >= theta
                    current_Q = None
                    current_Q_indices = []
                    
                    # Vectorized operations to avoid double for loop over neurons i and j in layers K and K+1
                    # Limit search to top 10% of neurons in the layer above (ignore low correlation values)
                    k = int(len(Q[t][0]) / 10)
                    current_Q_indices = torch.topk(Q[t][0], k, dim=0)[1][:, 0]

                    # Alternatively, only look at neurons with more than theta activation (can cause problems if none satisfy condition)
#                     current_Q_indices = torch.where(Q[t][0] > theta[layer + 1])[0].unique()
#                     current_Q_indices = torch.gt(Q[t][0], theta[layer + 1]).nonzero()[:, 0].unique()
#                     print("equal? ", current_Q_indices == current_Q_indices_2)

                    current_Q = generate_correlation_map(act_sample, act_next_sample[current_Q_indices])
                    
#                     print("Shape of j values: ", current_Q_indices.shape)
                    print("Shape of vectorized Q matrix: ", current_Q.shape)
                    
                    torch.save(current_Q_indices, f"Q{args.output_name}/class{t}layer{layer}_indices.pt" )
                    torch.save(current_Q, f"Q{args.output_name}/class{t}layer{layer}.pt") 
                    
                    Q[t].insert(0, current_Q.cpu())
                    Q_indices[t].insert(0, current_Q_indices.cpu())

                # Debugging: make sure cache is empty at the start of each iteration as we do not need to save anything in gpu from past iteration
                # Shouldn't need this though as they are all defined within a for loop...
                torch.cuda.empty_cache()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # Debugging: print time every iteration
                print("Batch time: ", time.time() - end)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                                
                if i % args.print_freq == 0:
                    progress.display(i)
        progress.display_summary()
             
        # --- Export activations & indices --  
        with open(os.getcwd() + '/output' + args.output_name + '/pred_activations.pickle', 'wb') as handle:
            pickle.dump(pred_activations, handle)
            
        with open(os.getcwd() + '/output' + args.output_name + '/indices.pickle', 'wb') as handle:
            pickle.dump(indices, handle)
        
        # TODO: Clear GPU memory to make sure no cuda memory errors for larger sizes
        
        # --- Prediction ---
        
        total_images = torch.zeros(num_layers)
        total_correct = torch.zeros(num_layers)
        total_correct_alt = torch.zeros(num_layers)
        acuracy = torch.zeros(num_layers)
        prediction = torch.zeros(len(pred_activations[0][0]) * N)
        prediction_alt = torch.zeros(len(pred_activations[0][0]) * N)

        print("Total images to predict: ", len(prediction))
        print("Predicting images...")

        # Define thresholds outside of loop to print later for easier debuggin
        threshold1 = 0
        thresdhold2 = 0
        
        # Predict images using each layer of the network (16 images from validation set)
        # t_star controls which image activations we are looking at, 16 per class (160 total)
        for t_star in range(N):
            print("Class ", t_star)
            for image_num in range(len(pred_activations[0][0])):
#                 print(f"--- IMAGE {image_num} of class {t_star}")
                S = torch.zeros((N, num_layers)) # Only need to calculate S for each image sequentially
                S_alt = torch.zeros((N, num_layers))

                for layer in range(num_layers):
        #             print("Layer ", layer)
                    for t in range(N):
                        # Multiply A * B where A is shape [M] and B is shape [M. N] for M subsampled neurons in layer K, N subsampled neurons in layer K+1
                        # Take sum of activation for each neuron multipled with Q values for that neuron and corresponding neurons in next layer

                        threshold1 = theta[layer].item()
                        S[t][layer] = torch.sum(
                                        torch.matmul(
                                            pred_activations[t_star][layer][image_num].to(device), 
                                            F.threshold(Q[t][layer].to(device), threshold1, 0)
                                            )
                                        )
                        threshold2 = 0
                        S_alt[t][layer] = torch.sum(
                                        torch.matmul(
                                            pred_activations[t_star][layer][image_num].to(device), 
                                            F.threshold(Q[t][layer].to(device), threshold2, 0)
                                            )
                                        )

                    # class * 16 + image_num
                    total_images_index = t_star * len(pred_activations[0][0]) + image_num
                    prediction[total_images_index] = torch.argmax(S[:, layer]) # Find the t value for which it is the highest
                    prediction_alt[total_images_index] = torch.argmax(S_alt[:, layer]) # Find the t value for which it is the highest

        #             print("Absolute value S values: ", S[:, layer])
        #             print("Regular S values       : ", S_alt[:, layer])
        #             print("Correct class: ", t_star)
        #             print("Predicted class: ", prediction[image_num])
        #             print("Predicted class (alternate): ", prediction_alt[image_num])

                    # We know the class is t for all the images in this batch, which makes it easy
                    total_images[layer] += 1
                    if int(prediction[total_images_index]) == t_star:
                        total_correct[layer] += 1
        #                 print("GOT ONE RIGHT")
                    if int(prediction_alt[total_images_index]) == t_star:
                        total_correct_alt[layer] += 1
        #                 print("GOT ONE RIGHT (ALT)")
        print(total_images)
        acc = total_correct / total_images
        acc_alt = total_correct_alt / total_images
        print(f"Accuracy (threshold={threshold1}): {acc}")
        print("TOTAL CORRECT: ", sum(total_correct))
        print(f"Accuracy (threshold={threshold2}): {acc_alt}")
        print("TOTAL CORRECT ALT: ", sum(total_correct_alt))
        np.save("output" + args.output_name + "/accuracy.npy", acc) # Layer by layer accuracy output
            
    
    return top1.avg

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : torch.tensor
      Shape N X T.

    y : torch.tensor
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' + 
                        'have the same number of timepoints.')
    if type(x) == np.ndarray and type(y) == np.ndarray:
        s_x = x.std(1, ddof=n - 1)
        s_y = y.std(1, ddof=n - 1)
        cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
        normalization = np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])
#         if np.any(normalization == 0):
#             print("-- Divide by zero ---")
#             x_zeros = np.where(s_x == 0)
#             y_zeros = np.where(s_y == 0)
#             print("Indices where s_x is zero: ", x_zeros)
#             print("Portion of x indices which are zero: ", len(x_zeros) / len(x))
#             print("Values over example images (34) where std of neuron is zero: ", x[x_zeros])
            
#             print("Indices where s_y is zero: ", y_zeros)
#             print("Portion of y indices which are zero: ", len(y_zeros) / len(y))
#             print("Values over example images (34) where std of neuron is zero: ", y[y_zeros])
        return cov / normalization
    if type(x) == torch.Tensor and type(y) == torch.Tensor:
        s_x = x.std(1, unbiased=True) * math.sqrt(n-1) # delta degrees of freedom n-1
        s_y = y.std(1, unbiased=True) * math.sqrt(n-1) # delta degrees of freedom n-1
        cov = torch.matmul(x, y.T) - n * torch.matmul(mu_x.unsqueeze(1), mu_y.unsqueeze(0))
        normalization = torch.matmul(s_x.unsqueeze(1), s_y.unsqueeze(0))
#         if torch.any(normalization== 0):
#             print("--- Divide by zero ---")
#             x_zeros = torch.where(s_x == 0)
#             y_zeros = torch.where(s_y == 0)
#             print("Indices where s_x is zero: ", x_zeros)
#             print("Portion of x indices which are zero: ", len(x_zeros) / len(x))
#             print("Values over example images (34) where std of neuron is zero: ", x[x_zeros])

#             print("Indices where s_y is zero: ", y_zeros)
#             print("Portion of y indices which are zero: ", len(y_zeros) / len(y))
#             print("Values over example images (34) where std of neuron is zero: ", y[y_zeros])
        return cov / normalization
    else:
        raise ValueError('x and y must be both either ndarrays or tensors')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class SaveOutput:
    """Saves output from intermediate layer activations"""
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach().cpu())
    
    def clear(self):
        self.outputs = []

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
