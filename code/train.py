import os, sys
import torch  
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pathlib

#script_location="/Users/can/Documents/GitHub"
up = lambda pth: pth.parent.resolve()

script_location = up(pathlib.Path(__file__))
module_location = up(up(script_location))

sys.path.append( str(module_location) )

from dlsisr.code.common.read_data import prepare_images_att

#path = '/Users/can/Documents/GitHub/dlsisr/data/ORL-DATABASE'
path = str(module_location / 'dlsisr' / 'data' / 'ORL-DATABASE')
seen_people_tr,seen_people_te,unseen_people=prepare_images_att(path)
#%%


from dlsisr.code.common.disruptor import add_blur_decrease_size

resize_dim=(64,64)

width,height,seen_people_tr=add_blur_decrease_size(seen_people_tr,resize_dim,add_blur=False)
width,height,seen_people_te=add_blur_decrease_size(seen_people_te,resize_dim,add_blur=False)
width,height,unseen_people=add_blur_decrease_size(unseen_people,resize_dim,add_blur=False)

#check that seen_people_tr from 0 to 8 will give us the same person.
#check that seen_people_te from 0 to 1 will give us the same person.
#check that unseen_people has 8 different people. Each person has 10 different images.
plt.imshow(seen_people_tr[0,],cmap="gray")
#plt.imshow(seen_people_te[0,],cmap="gray")
#plt.imshow(unseen_people[0,],cmap="gray")


desired_dim=(16,16)

width,height,seen_people_tr_low_qual=add_blur_decrease_size(seen_people_tr,desired_dim,add_blur=False)
width,height,seen_people_te_low_qual=add_blur_decrease_size(seen_people_te,desired_dim,add_blur=False)
width,height,unseen_people_low_qual=add_blur_decrease_size(unseen_people,resize_dim,add_blur=False)
plt.imshow(seen_people_tr_low_qual[0,],cmap="gray")

resize_dim=(64,64)
width,height,interpolate_seen_people_tr=add_blur_decrease_size(seen_people_tr_low_qual,resize_dim,add_blur=False)
width,height,interpolate_seen_people_te=add_blur_decrease_size(seen_people_te_low_qual,resize_dim,add_blur=False)
width,height,interpolate_unseen_people=add_blur_decrease_size(unseen_people_low_qual,resize_dim,add_blur=False)
plt.imshow(interpolate_seen_people_tr[0,],cmap="gray")



#%%

from dlsisr.code.common.plots import draw_triplet
image_num=0
#note that we did not predict any image yet. Need to set this to the image
#generated by the nn.
predicted=interpolate_seen_people_tr[image_num]
draw_triplet(predicted,seen_people_tr[image_num],interpolate_seen_people_tr[image_num])


