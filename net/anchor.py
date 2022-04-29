import torch
from d2l import torch as d2l
# import sys
# sys.path.append('..')


def multibox_prior(data, sizes, ratios):
    """[Product anchors with different shape and size]
    anchor width = width * s * sqrt(r)
    anchor height = height * s / sqrt(r)
    anchor number:(s1, r1),(s1, r2), . . . ,(s1, rm),(s2, r1),(s3, r1), . . . ,(sn, r1)
    Args:
        data ([type]): [the feature map]
        sizes ([type]): [the size ratio]
        ratios ([type]): [the ratio between width and height]
    Returns:
    output.unsqueeze(0): the matrix of anchors for every pixel 1 * (h * W * num_anchors) * 4    
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios -1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    
    # Offset is set to move anchor point to the center of pixel. 
    # The offset is 0.5 because of the pixel size 1 * 1. 
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # Scaled steps in y axies
    steps_w = 1.0 / in_width # Scaled steps in x axies
    
    # Product all anchor center. 
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h 
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w 
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    # Product 'boxes_per_pixel' height and width. 
    # Then product the four corner axle (xmin, xmax, ymin, ymax). 
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width # Handle rectangular inputs.
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    # Half height and width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    
    # Every center will have 'boxes_per_pixel' number of anchors.  
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

