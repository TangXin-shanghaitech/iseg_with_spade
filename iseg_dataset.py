from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import matplotlib.pyplot as plt
import imageio
from nibabel.funcs import squeeze_image
import numpy as np
import torchvision.transforms as transforms
import torch
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import torch.nn.functional as F



class IsegDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(crop_size=256)
        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.label_dir = opt.label_dir
        self.image_dir = opt.image_dir
        self.instance_dir = opt.instance_dir
       
    def __getitem__(self, index):
        #label
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        fnames = os.listdir(self.label_dir)
        fnames.sort(key=lambda x: int(x.split('.')[0].split('-')[1])) 
        f = fnames[index]
        label_path = os.path.join(self.label_dir,f)
        label_data = np.load(label_path)
        label_data = torch.from_numpy(label_data)
        label_data = torch.unsqueeze(label_data,dim=0)
        pad_dims = (        
                            32,32,
                            56,56,
                            # 32,32,
                                )
        label_data=F.pad(label_data,pad_dims,"constant") 

        # print("label shape",label_data.shape)
        

        #img
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        fnames = os.listdir(self.image_dir)

        fnames.sort(key=lambda x: int(x.split('.')[0].split('-')[1])) 
        f = fnames[index]
        img_path = os.path.join(self.image_dir,f)
        img_data = np.load(img_path)
        img_data = torch.from_numpy(img_data)
        img_data = torch.stack((img_data,img_data,img_data),dim=0)
        img_data = img_data.float()
        img_data = img_data/torch.max(img_data)
        img_data = transform(img_data)
        pad_dims = (        
                            32,32,
                            56,56,
                            # 32,32,
                                )
        img_data=F.pad(img_data,pad_dims,"constant") 
        # print("img shape",img_data.shape)

        #instance
        
        ins_data = 0

        input_dict = {'label': label_data,
                      'instance': ins_data,
                      'image': img_data,
                      'path': img_path,
                      }

        return input_dict

        


    def __len__(self):
        f = os.listdir(self.label_dir)
        return len(f)
    

        


