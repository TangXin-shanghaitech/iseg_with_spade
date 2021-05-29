import nibabel as nib
import matplotlib.pyplot as plt
import os
import imageio  #写入图片
from nibabel.funcs import squeeze_image
import numpy as np
from PIL import Image  #读取图片
import torchvision.transforms as transforms
import torch


t1path = r"C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training\subject-1-T1.img"
path = r'C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training'
imgpath = r'C:\Users\tangx\Desktop\new\img'
labelpath = r'C:\Users\tangx\Desktop\new\label'
#get the 3-Ddata
def read_data(path):
    img = nib.load(path)
    image_data = img.get_data()
    
    image_data = np.squeeze(image_data)

    return image_data


def slice(img_data):
    img_data = img_data[:,:,150]
    img_data = np.squeeze(img_data)
    return img_data


def niitonpy(path_to_all_niifile,path_to_img,path_to_label):
    filesnames = os.listdir(path_to_all_niifile)
    for fname in filesnames:

        if 'T1.img' in fname:
            continue
            print("文件名",fname)
            fpath = os.path.join(path_to_all_niifile,fname) 
            print("..............文件路径................",fpath)
            fname = fname.split('.')[0]
            newfpath = os.path.join(path_to_img,fname)
            print(".............存储路径.............",newfpath)
            img_data = read_data(fpath)
            img_data = slice(img_data)
            print("type(img_data)",type(img_data))
            print("img_data.shape",img_data.shape)
            np.save(newfpath,img_data)
        if 'label.img' in fname:
            fpath = os.path.join(path_to_all_niifile,fname) 
            fname = fname.split('.')[0]
            newfpath = os.path.join(path_to_label,fname)
            img_data = read_data(fpath)
            img_data = slice(img_data)
            np.save(newfpath,img_data)


# niitonpy(path,imgpath,labelpath) 
            
#判断是否相等
img_data = np.load(r'C:\Users\tangx\Desktop\new\label\subject-1-label.npy')
raw_data = read_data(r'C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training\subject-1-label.img')
raw_data = slice(raw_data)

judge = (img_data == raw_data)
print('...........................................................................',judge.all())
            


            
            



    
# a = np.load(r'C:\Users\tangx\Desktop\new\out.npy')
# print(a.shape)

# img_data = read_data(t1path)
# img_data = slice(img_data)
# np.save(r"C:\Users\tangx\Desktop\new\subject-1-T1",img_data)   







# print(type(raw_imgdata))    numpy

# new_imgdata = np.load(savepath)






# def  show_img(ori_img):
#     plt.imshow(ori_img[:,:,150],cmap='gray')
#     plt.show()

# def nii_to_image(niifile,photofile):
#     filesnames = os.listdir(niifile)
#     slice_trans = []

#     for f in filesnames:  # f is subject-1-label.img

#         if '.img' in f:
            
#             img_path = os.path.join(niifile,f)
#             img = nib.load(img_path)
#             img_fdata = img.get_fdata()
#             fname = f.replace('.img','')
#             photo_path = photofile
            
#             img_fdata = np.squeeze(img_fdata)
#             (x,y,z) = img_fdata.shape
#             for i in range(140,180):
#                 slice = img_fdata[:,:,i]
#                 print(os.path.join(photo_path,'{}.png'.format(i)))
#                 imageio.imwrite(os.path.join(photo_path,'{}-{}.png'.format(fname,i)),slice)
#                 break   
#         print('ok')
        
# # nii_to_image(r'C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training',r'C:\Users\tangx\Desktop\new')

# #图片----tensor
# img = Image.open(r'C:\Users\tangx\Desktop\new\subject-1-label-140.png')  
# img = img.convert('RGB')
# transform = transforms.ToTensor()
# img_tensor = transform(img)
# img_tensor1 = img_tensor[1,:,:]
# img_tensor1.squeeze(0)
# img_tensor2 = img_tensor[2,:,:]
# img_tensor1.squeeze(1)
# print('.................................. img1.shape',img_tensor1.shape)

# #原始的数据  
# img_data = read_data(path)
# rimg_data = img_data[:,:,140]
# rimg_data_tensor = torch.from_numpy(rimg_data) 
# print('....................................rimg.shape',rimg_data_tensor.shape) 



# judge = (img_tensor1 == rimg_data_tensor)
# print('........................................................................................judge',judge.all())







