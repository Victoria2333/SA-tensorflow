import numpy as np
import pdb
import sys
import caffe
import os
import scipy.io as sio
import cPickle
import subprocess
import cv2

if os.path.isfile('/home/sxs4337/caffe_models/ResNet-152-model.caffemodel'):
    print 'CaffeNet found.'

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

model_def = '/home/sxs4337/caffe_models/ResNet-152-deploy.prototxt'
model_weights = '/home/sxs4337/caffe_models/ResNet-152-model.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.array((104, 117, 123))
print 'mean-subtracted values:', zip('BGR', mu)
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel

data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
batchsize = 1
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

f = open('dict_youtube_mapping.pkl', 'rb')
vid_map = cPickle.load(f)
f.close()
train = ['vid%s'%i for i in range(1,1201)]
val = ['vid%s'%i for i in range(1201,1301)]
test = ['vid%s'%i for i in range(1301,1971)]
seq_len = 50 # no. of frames per video
count = 0;

f = open('resnet_feat_all_pool5.pkl','w')
data = {}
content = []
for split in ['train', 'val', 'test']:
    files = 'list_of_'+split+'_images.txt'
    content.append(open(files).read().splitlines())
content1 = [item for sublist in content for item in sublist]
all_vid = ['vid%s'%i for i in range(1,1971)]
for i in all_vid:
    data[i] = np.array([])
for k in xrange(0,len(content1)):
    count += 1
    video_name = content1[k].split('/')[-2]
    file_name = content1[k]
    image = caffe.io.load_image(file_name)
    image = cv2.resize(image, (256, 256))
    width, height,channels = image.shape   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    cImg = image[left:right, top:bottom, :] # center crop
    transformed_image = transformer.preprocess('data', cImg)
    batch_image = transformed_image[...,np.newaxis]
    batch_image = np.transpose(batch_image,(3,0,1,2))
    net.blobs['data'].data[...] = batch_image
    output = net.forward()
    batch_feat = net.blobs['pool5'].data
    batch_feat = np.reshape(batch_feat,(1,2048))
    if data[video_name].size:
        data[video_name] = np.vstack((data[video_name],batch_feat))
    else:
        data[video_name] = batch_feat
    print "Done image %d of %d" % (count, len(content1))
cPickle.dump(data,f)
f.close()

pdb.set_trace()
