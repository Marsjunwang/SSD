from d2l import torch as d2l
import os
import pandas as pd
import torchvision
import torch

def read_data_bananas(is_train=True):
    """读取⾹蕉检测数据集中的图像和标签。"""
    # data_dir = d2l.download_extract('banana-detection') # download the data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_fname = os.path.join(data_dir, 'banana-detection','bananas_train' if is_train
                             else 'bananas_val','label.csv')
    if not os.path.exists(csv_fname):
        raise ValueError(f'Unknown folder path:{csv_fname} ')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'banana-detection', 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
    
class BananasDataset(torch.utils.data.Dataset):
    """One of dataset for loading the banana dataset"""   
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if 
              is_train else f' validation examples'))
    
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    
    def __len__(self):
        return len(self.features)
    

def load_data_bananas(batch_size):
    """加载⾹蕉检测数据集。"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter  

# batch_size, edge_size = 32, 256
# train_iter, _ = load_data_bananas(batch_size)
# batch = next(iter(train_iter))
# print(batch[0].shape, batch[1].shape)

# imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
# axes = d2l.show_images(imgs, 2, 5, scale=2)
# for ax, label in zip(axes, batch[1][0:10]):
#     d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
# d2l.plt.show()  
# print('1')
  
