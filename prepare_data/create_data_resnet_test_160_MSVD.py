import numpy as np
import cPickle
import pdb
import h5py
from random import shuffle
import os
import string
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
printable = set(string.printable)
fs = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]

test100 = ['vid%s'%i for i in range(1301,1971)]
max_captions = max([len(vid_capt[x]) for x,y in vid_map.iteritems() if y in test100])

for split in ['test100']:
  path1 = '/shared/kgcoe-research/mil/video_project/MSVD/h5_125_resnet_corr/'+split
  if not os.path.exists(path1):
    os.makedirs(path1)
  print "Working on split:", split
  split_files = eval(split)
  no_batches = len(split_files)/batch_size
  idx = range(0,len(split_files))
  count = 0
  for i in range(0,len(split_files),batch_size):
    b_fname = []
    b_title = []
    token_captions_all = []
    last = i+batch_size
    if i+batch_size > len(split_files):
      last = len(split_files)
    batch_idx = range(i,last)
    all_feat = np.array([])
    for j in batch_idx:
      b_fname.append(split_files[j])
      features = vid_feat[split_files[j]]
      features.astype('float32')
      
      caption_vid = vid_capt[[x for x,y in vid_map.iteritems() if y == split_files[j]][0]]
      caption_vid.extend(['']*(max_captions-len(caption_vid)))
      current_captions = {}
      current_captions['annotations'] = map(lambda x:{'image_id':0,'caption':x}, [i for i in caption_vid])
      token_captions = tokenizer.tokenize(current_captions)
      token_captions_all.append(token_captions['annotations'])
      
      if features.shape[0] < seq_len:
	    features_sub = np.vstack([features,np.zeros([seq_len-features.shape[0],feat_size])])
      else:
        #features_sub = features[0:seq_len, :]
        features_sub = features[fs(seq_len,features.shape[0]), :]
      if all_feat.size:
        all_feat = np.vstack((all_feat,features_sub))
      else:
        all_feat = features_sub
      new_caption = []
      # for sent in caption_vid:
        # new_sent = filter(lambda x:x in printable,sent)
        # new_caption.append(new_sent)
      # b_title.append(new_caption)
    all_feat = all_feat.reshape((seq_len,len(batch_idx),feat_size),order='F')
    b_data = all_feat
    b_label = -1*np.ones([seq_len,len(batch_idx)])
    b_label[-1,:] = 0
    count += 1
    with h5py.File(path1+'/'+'test'+str(count)+'.h5', 'w') as hf:
      hf.create_dataset('data', data=b_data)
      hf.create_dataset('fname', data=b_fname)
      hf.create_dataset('label', data=b_label)
      hf.create_dataset('title', data=token_captions_all)
pdb.set_trace()