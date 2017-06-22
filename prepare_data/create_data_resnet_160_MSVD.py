import numpy as np
import cPickle
import pdb
import h5py
from random import shuffle
import os
from ptbtokenizer import PTBTokenizer
tokenizer = PTBTokenizer()

f = open('/shared/kgcoe-research/mil/video_project/MSVD/resnet_feat_all_pool5.pkl', 'rb')
vid_feat = cPickle.load(f)
f.close()
f = open('dict_movieID_caption.pkl', 'rb')
vid_capt = cPickle.load(f)
f.close()
f = open('dict_youtube_mapping.pkl', 'rb')
vid_map = cPickle.load(f)
f.close()

batch_size = 100
seq_len = 125
feat_size = 2048
fs = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
train100 = ['vid%s'%i for i in range(1,1201)]
val100 = ['vid%s'%i for i in range(1201,1301)]
test = ['vid%s'%i for i in range(1301,1971)]

for split in ['train100']: #'val100'
  all_fname = []
  all_captions = []
  all_labels = np.array([])
  print "Working on split:", split
  split_files = eval(split)
  for k, v in vid_map.items(): # k is video name, v is video number
    if v in split_files: # check if video in correct split
      captions = vid_capt[k]
      for cur_caption in captions:
        all_fname.append(v)
        all_captions.append(cur_caption)
        # labels = -1*np.ones([seq_len,1])
        # labels[-1,0] = 0
        # if all_labels.size:
          # all_labels = np.vstack((all_labels,labels))
        # else:
          # all_labels = labels
  if not os.path.exists('/shared/kgcoe-research/mil/video_project/MSVD/h5_125_resnet_corr/'+split):
    os.makedirs('/shared/kgcoe-research/mil/video_project/MSVD/h5_125_resnet_corr/'+split)
  no_batches = len(all_fname)/batch_size
  idx = range(0,len(all_fname))
  shuffle(idx)
  print "No. of samples:", len(all_fname)
  print "No. of batches:", no_batches
  all_labels = all_labels.reshape((seq_len,-1),order='F')

  for batch in range(1,no_batches+1):
    batch_idx = idx[batch_size*(batch-1):batch_size*batch]
    b_title = [all_captions[i] for i in batch_idx]
    b_fname = [all_fname[i] for i in batch_idx]
    # b_label = all_labels[:,batch_idx]
    b_label = -1*np.ones([seq_len,batch_size])
    b_label[-1,:] = 0
    # create data batch
    all_feat = np.array([])
    #pdb.set_trace()
    for feat_vid in b_fname:
      features = vid_feat[feat_vid]
      if features.shape[0] < seq_len:
        features_sub = np.vstack([features,np.zeros([seq_len-features.shape[0],feat_size])]) # zero padding
      else:
        #features_sub = features[0:seq_len, :] # 125 equaly spaced fs(seq_len,features.shape[0])
        features_sub = features[fs(seq_len,features.shape[0]), :]
      #features_sub = features[np.linspace(0,features.shape[0]-1,seq_len,dtype=int),:]
      features_sub.astype('float32')
      if all_feat.size:
        all_feat = np.vstack((all_feat,features_sub))
      else:
        all_feat = features_sub
    all_feat = all_feat.reshape((seq_len,batch_size,feat_size),order='F')
    b_data = all_feat

    token_captions_all = []
    current_captions = {}
    current_captions['annotations'] = [{'image_id':0,'caption':cap} for cap in b_title]
    token_captions = tokenizer.tokenize(current_captions)
    token_captions_all.append(token_captions['annotations'])

    with h5py.File('/shared/kgcoe-research/mil/video_project/MSVD/h5_125_resnet_corr/'+split+'/'+split+str(batch)+'.h5', 'w') as hf:
      hf.create_dataset('data', data=b_data)
      hf.create_dataset('fname', data=b_fname)
      hf.create_dataset('label', data=b_label)
      hf.create_dataset('title', data=token_captions_all[0])
