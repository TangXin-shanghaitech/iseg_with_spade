from posixpath import split
import nibabel as nib
import matplotlib.pyplot as plt
import os
import imageio  #写入图片
from nibabel.funcs import squeeze_image
import numpy as np
from PIL import Image  #读取图片
import torchvision.transforms as transforms
import torch
import re



t1path = r"C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training\subject-1-T1.img"
path = r'C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training'


ade20label = r'C:\Users\tangx\Documents\GitHub\SPADE\datasets\ADE20K\ADEChallengeData2016\annotations\training\ADE_train_00000001.png'
ade20img = r'C:\Users\tangx\Documents\GitHub\SPADE\datasets\ADE20K\ADEChallengeData2016\images\training\ADE_train_00000001.jpg'

labelnpy_path = r'C:\Users\tangx\Desktop\new\label'
imgnpy_path = r'C:\Users\tangx\Desktop\new\img'



def get_img_to_tensor(imgspath,index):

    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    fnames = os.listdir(imgspath)
    # path_list.sort(key=lambda x:int(x.split('img_')[1].split('.txt')[0]))
    fnames.sort(key=lambda x: int(x.split('.')[0].split('-')[1])) 
    f = fnames[index]
    img_path = os.path.join(imgspath,f)
    img_data = np.load(img_path)
    img_data = torch.from_numpy(img_data)
    img_data = torch.stack((img_data,img_data,img_data),dim=0)
    img_data = img_data.float()
    img_data = img_data/torch.max(img_data)
    img_data = transform(img_data)

    
    # img_data = transform(img_data)
    # img_data = img_data * 255
    return img_data
    

    


a = get_img_to_tensor(imgnpy_path,9)

numlist = []
for k in range(3):
    for i in range(144):
        for j in range(192):
            if a[k,i,j].item() not in numlist:
                numlist.append(a[k,i,j].item())
print('.............................................................',numlist)
print('     .....................................................a.dtype',a.dtype)
print('.....................................................a',a)
# f =  'subject-2-T1.npy'
# a = re.split("[.-]",f)[1]
# print(a)



image = Image.open(r'C:\Users\tangx\Documents\GitHub\SPADE\datasets\ADE20K\ADEChallengeData2016\images\training\ADE_train_00000001.jpg')   
image = image.convert('RGB') #
transform1 = transforms.ToTensor()  #0-1
transform2 = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                               
image = transform1(image)
image = transform2(image)


print('........................................................................................',image)
print("                            .........................................................image.dtype",image.dtype)






# def get_label_to_tensor(labelspath,index):
#     fnames = os.listdir(labelspath)
#     for f in fnames:
#         if index in f:
#             label_path = os.path.join(labelspath,f)
#             label_data = np.load(label_path)
#             label_data = torch.from_numpy(label_data)
#             label_data = torch.unsqueeze(label_data,dim=0)
#             return label_data




# label_data = get_label_to_tensor(labelnpy_path,"2")
            
# numlist = []
# for k in range(1):
#     for i in range(144):
#         for j in range(192):
#             if label_data[k,i,j].item() not in numlist:
#                 numlist.append(label_data[k,i,j].item())
# print(numlist)






# get_label(labelnpy_path,"2")
















#get the 3-Ddata
# def read_data(path):
#     img = nib.load(path)
#     image_data = img.get_data()
    
#     image_data = np.squeeze(image_data)

#     return image_data


# def slice(img_data):
#     img_data = img_data[:,:,150]
#     img_data = np.squeeze(img_data)
#     return img_data


# def niitonpy(path_to_all_niifile,path_to_img,path_to_label):
#     filesnames = os.listdir(path_to_all_niifile)
#     for fname in filesnames:

#         if 'T1.img' in fname:
            
#             print("文件名",fname)
#             fpath = os.path.join(path_to_all_niifile,fname) 
#             print("..............文件路径................",fpath)
#             fname = fname.split('.')[0]
#             newfpath = os.path.join(path_to_img,fname)
#             print(".............存储路径.............",newfpath)
#             img_data = read_data(fpath)
#             img_data = slice(img_data)
#             print("type(img_data)",type(img_data))
#             print("img_data.shape",img_data.shape)
#             np.save(newfpath,img_data)
#         if 'label.img' in fname:

#             fpath = os.path.join(path_to_all_niifile,fname) 
#             fname = fname.split('.')[0]
#             newfpath = os.path.join(path_to_label,fname)
#             img_data = read_data(fpath)
#             img_data = slice(img_data)
#             np.save(newfpath,img_data)
# niitonpy(path,imgpath,labelpath)





















# label = Image.open(ade20label)
# label = label.convert('RGB')
# transform = transforms.ToTensor()
# label_tensor = transform(label)
# label_tensor = label_tensor * 255.0
# label_tensor = label_tensor.squeeze(0)
# numlist = []
# # for i in range(512):
# #     for j in range(683):
# #         if label_tensor[i,j].item() not in numlist:
# #             numlist.append(label_tensor[i,j].item())
# # print(numlist)

# # img = Image.open(ade20img)
# # transform = transforms.ToTensor()
# # img_tensor = transform(img)
# # img_tensor = img_tensor * 255.0
# # img_tensor = img_tensor.squeeze(0)
# # numlist = []
# # for i in range(512):
# #     for j in range(683):
# #         if label_tensor[i,j].item() not in numlist:
# #             numlist.append(label_tensor[i,j].item())
# # # print(numlist)
# for k in range(3):
#     for i in range(512):
#         for j in range(683):
#             if label_tensor[k,i,j].item() not in numlist:
#                 numlist.append(label_tensor[k,i,j].item())
# print(numlist)

# print('..................................................label_tensor.shape',label_tensor.shape)
# print('..................................................label_tensor.type',type(label_tensor))

# print('..................................................label_tensordtype',label_tensor.dtype)





# print('..................................................label_tensor.shape',img_tensor.shape)
# print('..................................................label_tensor.type',type(img_tensor))

# print('..................................................label_tensordtype',img_tensor.dtype)
















# niitonpy(path,imgpath,labelpath) 
            
#判断是否相等
# img_data = np.load(r'C:\Users\tangx\Desktop\new\label\subject-1-label.npy')
# raw_data = read_data(r'C:\Users\tangx\Desktop\project\iSeg2019\iSeg-2019-Training\subject-1-label.img')
# raw_data = slice(raw_data)

# judge = (img_data == raw_data)
# print('...........................................................................',judge.all())
            


            
            



    
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







