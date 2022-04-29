#!/usr/bin/python3

from operator import concat
import torch
import torchvision
from torch import nn
from torch import functional as F  
from d2l import torch as d2l

from net.base import get_blk
from net.predictor import cls_predictor
from net.predictor import bbox_predictor
from net.anchor import multibox_prior
from viewer.bbox_visual import show_bboxes
from data.read_data import load_data_bananas
from loss.loss import calc_loss, cls_eval, bbox_eval

def flatten_pred(pred):
    """[summary]

    Args:
        pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """This is a base module forward functgetattr
        X ([matrix]): [The input feature map]
        blk ([function]): [One of calculation function of the module]
        size ([float32]): [The anchor size]
        ratio ([type]): [The anchor ratio]
        cls_predictor ([type]): [class predictor]
        bbox_predictor ([type]): [bounbox predictor]

    Returns:
        Y[matrix]: [Feature map]
        anchors:
        cls_preds:[classification]
        bbox_pres:[offset]
    """
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class TinySSD(nn.Module):
    """A single shot multi-frame object detection network

    Attributes:
        num_classes ([int]): the number of classfication
    """
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # assignment statement 'self.blk_i = get_blk(i)'
            setattr(self,f'blk_{i}',get_blk(i))
            setattr(self,f'cls_{i}',cls_predictor(idx_to_in_channels[i],
                                                  num_anchors, num_classes))
            setattr(self,f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                     num_anchors))
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            #getattr(self, 'blk_%d' % i) equal to use 'self.blk_i'
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


if __name__ == "__main__":
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1 # The number of anchors for every
    anchor_test = False  
    if anchor_test:
        d2l.set_figsize()
        img = d2l.plt.imread('./net/catdog.jpg')
        h, w = img.shape[:2]
        print(h, w)
        X = torch.rand(size=(1, 3, h, w))
        Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
        Y.shape       
        boxes = Y.reshape(h, w , 5, 4)
        print(boxes[250, 250, 0, :])
        
        bbox_scale = torch.tensor((w, h, w, h))
        fig = d2l.plt.imshow(img)
        show_bboxes(fig.axes, boxes[250, 100, :, :] * bbox_scale,
                    ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                     's=0.75, r=0.5'])
        d2l.plt.show()
    batch_size = 32
    train_iter, _ = load_data_bananas(batch_size)
    device, net = d2l.try_gpu(), TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    
    num_epochs, timer = 20, d2l.Timer() 
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['class error', 'bbox mae'])
    net = net.to(device)
    for epoch in range(num_epochs):
        # Train precision sum,
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # Product different size anchor and predict the cls and offset for every anchor
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classification and offset
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            # Calculate nm the loss
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
         
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on'
              f'{str(device)}')
    d2l.plt.show()
           
    
        