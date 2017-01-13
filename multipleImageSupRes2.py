from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.objectives import squared_error
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import math
import scipy.misc
import scipy.io as sio
import string
import h5py
import time

from hashlib import sha1

from numpy import all, array, uint8


from PIL import Image

import sys
from os import listdir
from os.path import isfile, join

sys.path.insert(0,'/home/titanx1/Downloads/caffe-master/python')
import caffe

net_caffe = caffe.Net('VDSR_net.prototxt', '_iter_VDSR_Official.caffemodel', caffe.TEST)
layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))


TEST_MODE = False

SCALE_BASE = 1.1

SEQ_LEN = 4

# Optimization learning rate
LEARNING_RATE = 0.01#0.    00001

# All gradients above this will be clipped
GRAD_CLIP = 0.01

# Number of epochs to train the net
NUM_EPOCHS = 5000

PATCH_SIZE = 41

BATCH_SIZE = 30

STRIDE = 18


#Lasagne Seed for Reproducibility
#lasagne.random.set_rng(np.random.RandomState(1))


class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped
        

class MyObjective():
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(MyObjective, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss + 0.01 * lasagne.regularization.l2(self.input_layer)
        else:
            return loss

	
def load_dataset(mypath):	
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    N = 0
    for f in onlyfiles:
        img = Image.open(join(mypath, f))
        W,H = img.size
        for r in xrange(0,H-PATCH_SIZE,STRIDE):
            for c in xrange(0,W-PATCH_SIZE,STRIDE):
                N= N + 1
    seqBatch = np.zeros((N,SEQ_LEN,PATCH_SIZE,PATCH_SIZE))
    seqBatchlabels = np.zeros((N,1,PATCH_SIZE,PATCH_SIZE))
    for k in xrange(SEQ_LEN):
         n = 0
         for f in onlyfiles:
             img = Image.open(join(mypath, f)).convert('RGB')
             ycbcr = img.convert('YCbCr')
             W,H = ycbcr.size
             highRes = np.array(ycbcr)*1.0/255
             ycbcr = ycbcr.resize((int(W*1.0/3),int(H*1.0/3)),resample=PIL.Image.LANCZOS)
             W1,H1 = ycbcr.size
             lowRes = ycbcr.resize((int(W1*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1))),int(H1*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1)))), resample=PIL.Image.LANCZOS)
             lowRes = 1.0/255 * np.array(lowRes.resize((W,H), resample=PIL.Image.LANCZOS))
             for r in xrange(0,H-PATCH_SIZE,STRIDE):
                 for c in xrange(0,W-PATCH_SIZE,STRIDE):		
                    seqBatch[n,k,:,:] = lowRes[r:r+PATCH_SIZE,c:c+PATCH_SIZE,0]
                    seqBatchlabels[n,0,:,:] = highRes[r:r+PATCH_SIZE,c:c+PATCH_SIZE,0]
                    n = n + 1	

    order = np.random.permutation(n)
    seqBatch = seqBatch[order,:,:,:]
    seqBatchlabels = seqBatchlabels[order,:,:,:]
    seqBatch = seqBatch.astype(np.float32)
    seqBatchlabels = seqBatchlabels.astype(np.float32)
    return seqBatch,seqBatchlabels
	


def genBatchAndSequence(patchSet,p):
	imgBatch = patchSet[p:p+BATCH_SIZE,:,:]	
	seqBatch = np.zeros((BATCH_SIZE,SEQ_LEN,PATCH_SIZE,PATCH_SIZE))
	seqBatchlabels = np.zeros((BATCH_SIZE,SEQ_LEN,PATCH_SIZE,PATCH_SIZE))
	for i in xrange(BATCH_SIZE):
		img = imgBatch[i,:,:]
		lowbasedres = scipy.misc.imresize(img,1.0/3)
		for k in xrange(SEQ_LEN):
			lowRes = scipy.misc.imresize(scipy.misc.imresize(lowbasedres,1.0/(SCALE_BASE**(SEQ_LEN -  k- 1))),[PATCH_SIZE,PATCH_SIZE])			
			seqBatch[i,k,:,:] = lowRes*1.0/255
			seqBatchlabels[i,k,:,:] = img*1.0/255
	return seqBatch.reshape(SEQ_LEN*BATCH_SIZE,1,PATCH_SIZE,PATCH_SIZE),seqBatchlabels.reshape(SEQ_LEN*BATCH_SIZE,1,PATCH_SIZE,PATCH_SIZE)
	

    	
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype(np.float) - min_val) / (max_val - min_val)
    return out
    
def load_hdf5_data():
    
    with h5py.File('./preprocessing_code/mul_image_train.h5','r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = data[::2,:,:,:]
        label = label[::2,:,:,:]        
    return data,label
    
def main(num_epochs = NUM_EPOCHS):

	 # Load the dataset
    print("Loading data...")
    #X_train, y_train =  load_dataset("91")
    X_train, y_train =  load_hdf5_data()
    y_train = y_train[:,SEQ_LEN-1,:,:]
    [a,b,c] = y_train.shape
    y_train = y_train.reshape([a,1,b,c])
    
#==============================================================================
#    X_train = X_train.astype(np.uint8)
#    y_train = y_train.astype(np.uint8)
#    print(y_train.shape)
#    N= X_train.shape[0]
#    idx = np.random.randint(1,N,10)#
#    for i in idx:
#        print(y_train[i,0,:,:])
#        scipy.misc.imsave("high_"+ str(i)+"_" + str(0) + ".png",y_train[i,0,:,:])
#        for j in xrange(1):
#            scipy.misc.imsave("low_"+ str(i)+"_" + str(j) + ".png",X_train[i,j,:,:])
#==============================================================================
	# create Theano variables for input and target minibatch
    X_train = X_train/255.0
    y_train = y_train/255.0
    
    input_values = T.tensor4('X')
	#target_var = T.tensor4('Y')

	# construct CNN net
    net = {}
    net1={}
    net2={}
    net3={}
    net4={}
	#net['input'] = lasagne.layers.InputLayer((None, 1, PATCH_SIZE, PATCH_SIZE),input_values)
#    if TEST_MODE:
#        net['input'] = lasagne.layers.InputLayer((None, SEQ_LEN, None, None),input_values)
#    else:
#        net['input'] = lasagne.layers.InputLayer((None, SEQ_LEN, PATCH_SIZE, PATCH_SIZE),input_values)	
#            
    inputlayer = lasagne.layers.InputLayer((None, SEQ_LEN, None, None),input_values)
#==============================================================================
#     net['slice1'] = lasagne.layers.SliceLayer(net['input'], indices=slice(0, 1), axis=1)
#     net['slice2'] = lasagne.layers.SliceLayer(net['input'], indices=slice(1, 2), axis=1)
#     net['slice3'] = lasagne.layers.SliceLayer(net['input'], indices=slice(2, 3), axis=1)
#     net['slice4'] = lasagne.layers.SliceLayer(net['input'], indices=slice(3, 4), axis=1)
#==============================================================================
    net1['input']  = lasagne.layers.SliceLayer(inputlayer, indices=slice(0, 1), axis=1)
    net2['input']  = lasagne.layers.SliceLayer(inputlayer, indices=slice(1, 2), axis=1)
    net3['input']  = lasagne.layers.SliceLayer(inputlayer, indices=slice(2, 3), axis=1)
    net4['input']  = lasagne.layers.SliceLayer(inputlayer, indices=slice(3, 4), axis=1)
    

#==============================================================================
#     for i in xrange(SEQ_LEN):
#         namelayer = 'conv1_{}'.format(i+1)
#         name_slice_input = 'slice{}'.format(i+1)
#         net[namelayer]  = lasagne.layers.Conv2DLayer(net[name_slice_input],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
#         net[namelayer].W.tag.grad_scale = 0
#         net[namelayer].b.tag.grad_scale = 0.0
#==============================================================================
    net1['conv1']  = lasagne.layers.Conv2DLayer(net1['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
    net1['conv1'].W.tag.grad_scale = 1.0
    net1['conv1'].b.tag.grad_scale = 0.1
    net2['conv1']  = lasagne.layers.Conv2DLayer(net2['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=net1['conv1'].W,b=net1['conv1'].b) 
    #net2['conv1']  = lasagne.layers.Conv2DLayer(net2['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
    
    #net2['conv1'].W.tag.grad_scale = 1.0
    #net2['conv1'].b.tag.grad_scale = 0.1
    net3['conv1']  = lasagne.layers.Conv2DLayer(net3['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=net1['conv1'].W,b=net1['conv1'].b) 
    #net3['conv1']  = lasagne.layers.Conv2DLayer(net3['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
#   net3['conv1'].W.tag.grad_scale = 1.0
#   net3['conv1'].b.tag.grad_scale = 0.1
    net4['conv1']  = lasagne.layers.Conv2DLayer(net4['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=net1['conv1'].W,b=net1['conv1'].b) 
    #net4['conv1']  = lasagne.layers.Conv2DLayer(net4['input'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))        
#    net4['conv1'].W.tag.grad_scale = 1.0
#    net4['conv1'].b.tag.grad_scale = 0.1

    
    
#    for i in xrange(1,19):
#        for j in xrange(SEQ_LEN):
#            namelayer ='conv{}_{}'.format(i+1,j+1)
#            prvlayername = 'conv{}_{}'.format(i,j+1)
#            net[namelayer] =  lasagne.layers.Conv2DLayer(net[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#            if i>=0:
#                net[namelayer].W.tag.grad_scale = 0
#                net[namelayer].b.tag.grad_scale = 0.0
#            else:
#                net[namelayer].W.tag.grad_scale = 0
#                net[namelayer].b.tag.grad_scale = 0

    for i in xrange(1,19):
        namelayer ='conv{}'.format(i+1)
        prvlayername = 'conv{}'.format(i)
        net1[namelayer] =  lasagne.layers.Conv2DLayer(net1[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#        net2[namelayer] =  lasagne.layers.Conv2DLayer(net2[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#        net3[namelayer] =  lasagne.layers.Conv2DLayer(net3[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#        net4[namelayer] =  lasagne.layers.Conv2DLayer(net4[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))                
        net2[namelayer] =  lasagne.layers.Conv2DLayer(net2[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1[namelayer].W,b=net1[namelayer].b)   
        net3[namelayer] =  lasagne.layers.Conv2DLayer(net3[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1[namelayer].W,b=net1[namelayer].b)   
        net4[namelayer] =  lasagne.layers.Conv2DLayer(net4[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1[namelayer].W,b=net1[namelayer].b)
        net1[namelayer].W.tag.grad_scale = 1.0
        net1[namelayer].b.tag.grad_scale = 0.1
#        net2[namelayer].W.tag.grad_scale = 1.0
#        net2[namelayer].b.tag.grad_scale = 0.1
#        net3[namelayer].W.tag.grad_scale = 1.0
#        net3[namelayer].b.tag.grad_scale = 0.1
#        net4[namelayer].W.tag.grad_scale = 1.0
#        net4[namelayer].b.tag.grad_scale = 0.1
#      
    
#    for i in xrange(SEQ_LEN):
#        namelayer = 'conv20_{}'.format(i+1)
#        print(namelayer)
#        preNameLayer = 'conv19_{}'.format(i+1)
#        net[namelayer] = lasagne.layers.Conv2DLayer(net[preNameLayer],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.linear,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#        net[namelayer].W.tag.grad_scale = 0.0
#        net[namelayer].b.tag.grad_scale = 0.0

    net1['conv20'] = lasagne.layers.Conv2DLayer(net1['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#    net2['conv20'] = lasagne.layers.Conv2DLayer(net2['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.linear,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#    net3['conv20'] = lasagne.layers.Conv2DLayer(net3['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.linear,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#    net4['conv20'] = lasagne.layers.Conv2DLayer(net4['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.linear,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    
    net2['conv20'] = lasagne.layers.Conv2DLayer(net2['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1['conv20'].W,b=net1['conv20'].b)
    net3['conv20'] = lasagne.layers.Conv2DLayer(net3['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1['conv20'].W,b=net1['conv20'].b)
    net4['conv20'] = lasagne.layers.Conv2DLayer(net4['conv19'],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=net1['conv20'].W,b=net1['conv20'].b)
    
    net1['conv20'].W.tag.grad_scale = 1.0
    net1['conv20'].b.tag.grad_scale = 0.1
    net2['conv20'].W.tag.grad_scale = 1.0
    net2['conv20'].b.tag.grad_scale = 0.1
    net3['conv20'].W.tag.grad_scale = 1.0
    net3['conv20'].b.tag.grad_scale = 0.1
    net4['conv20'].W.tag.grad_scale = 1.0
    net4['conv20'].b.tag.grad_scale = 0.1
   
#    net1['sum']= lasagne.layers.ElemwiseSumLayer({net1['conv20'], net1['input']})
#    net2['sum']= lasagne.layers.ElemwiseSumLayer({net2['conv20'], net2['input']})
#    net3['sum']= lasagne.layers.ElemwiseSumLayer({net3['conv20'], net3['input']})
#    net4['sum']= lasagne.layers.ElemwiseSumLayer({net4['conv20'], net4['input']})
#
#    
#    net={} 
#    net['concat'] = lasagne.layers.ConcatLayer((net1['sum'],net2['sum'],net3['sum'],net4['sum']), axis = 1) 
#    net['conv21'] = lasagne.layers.Conv2DLayer(net['concat'],16,(3,3),pad = 1,nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#    net['conv22'] = lasagne.layers.Conv2DLayer(net['conv21'],1,(3,3),pad = 1,nonlinearity =  lasagne.nonlinearities.linear, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#==============================================================================

    net ={}
    net['concat'] = lasagne.layers.ConcatLayer((net1['conv20'],net2['conv20'],net3['conv20'],net4['conv20']), axis = 1) 
    net['conv21'] = lasagne.layers.Conv2DLayer(net['concat'],1,(3,3),pad = 1,nonlinearity =  lasagne.nonlinearities.linear, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    net['sum']= lasagne.layers.ElemwiseSumLayer({net['conv21'], net4['input']})

    print("Building network Multiple VDSR...")
    #l_out2 = lasagne.layers.ReshapeLayer(net['conv3'],(BATCH_SIZE*SEQ_LEN,1,c,PATCH_SIZE))
    l_out = net['sum']
    # Theano tensor for the targets
    target_values = T.tensor4('target_output')
    
    network_output = lasagne.layers.get_output(l_out)
#    network_output1 = lasagne.layers.get_output(net1['sum'])
#    network_output2 = lasagne.layers.get_output(net2['sum'])
#    network_output3 = lasagne.layers.get_output(net3['sum'])
#    network_output4 = lasagne.layers.get_output(net4['sum'])
    loss = 0.5*PATCH_SIZE**2*lasagne.objectives.squared_error(network_output, target_values).mean()
    #loss = 0.5*PATCH_SIZE**2*(squared_error(network_output[:,:,3:PATCH_SIZE-3,3:PATCH_SIZE-3], target_values[:,:,3:PATCH_SIZE-3,3:PATCH_SIZE-3]).mean())
    #+ 0.25*squared_error(network_output1, target_values).mean() + 0.25*squared_error(network_output2, target_values).mean()
    #+ 0.5*squared_error(network_output3, target_values).mean() + 0.5*squared_error(network_output4, target_values).mean())
    #MyObjective myobj
    #loss = MyObjective.get_loss(network_output, target_values).mean()
    # Retrieve all parameters from the network
    #params = lasagne.layers.get_all_params(l_out, trainable = True)
    # Compute AdaGrad updates for training
    print("Computing updates ...")
    #scaled_grads = lasagne.updates.total_norm_constraint(all_grads, 5)
    params = lasagne.layers.get_all_params(l_out)
    grads = theano.grad(loss, params)
    for idx, param in enumerate(params):
        grad_scale = getattr(param.tag, 'grad_scale', 1)
        if grad_scale != 1:
            grads[idx] *= grad_scale

    grads = theano.grad(loss, params)
    grads = [lasagne.updates.norm_constraint(grad, GRAD_CLIP, range(grad.ndim))
         for grad in grads]
    
    lr = theano.shared(np.array(LEARNING_RATE,dtype=theano.config.floatX))
    lr_decay = np.array(0.9,dtype=theano.config.floatX)
    updates = lasagne.updates.momentum(grads, params, learning_rate=lr,momentum=0.9)
    #updates = lasagne.updates.adagrad(grads, params, learning_rate=MYLEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([input_values, target_values], loss, updates = updates, allow_input_downcast = True)
    probs = theano.function([input_values], network_output, allow_input_downcast = True)
    
    #compute_cost = theano.function([input_values, target_values], loss, allow_input_downcast = True)
    #DATA_SIZE = X_train[0]
    
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================    
#    for item in net1.items():
#            name,layer =item 
#            if(string.find(name,'conv') >=0):
#                print(name)
#                print(layer)
#                W = layers_caffe[name].blobs[0].data[:,:,::-1,::-1]
#                b = layers_caffe[name].blobs[1].data
#                layer.W.set_value(np.transpose(W,(0,1,3,2)))
#                layer.b.set_value(b)
                #layer.W.set_value(layers_caffe[name[0:-2]].blobs[0].data[...,::-1,::-1])                
                #layer.b.set_value(layers_caffe[name[0:-2]].blobs[1].data)
                
#==============================================================================
           
        
    with np.load('MVDSR/official/MVDSR_5-softZ.npz') as fi:
          param_values = [fi['arr_%d' % i] for i in range(len(fi.files))]	
          lasagne.layers.set_all_param_values(l_out, param_values)
#    with np.load('MVDSR/official/MVDSR_65.npz') as fi:
#          param_values = [fi['arr_%d' % i] for i in range(len(fi.files))]	
#          for i in xrange(len(param_values) // 2):
#              if(i<20):
#                  name = 'conv{}'.format(i+1)
#                  net1[name].W.set_value(param_values[2*i])
#                  net1[name].b.set_value(param_values[2*i + 1])
#                  name = 'conv{}'.format(i+1)
#                  net2[name].W.set_value(param_values[2*i])
#                  net2[name].b.set_value(param_values[2*i + 1])
#                  name = 'conv{}'.format(i+1)
#                  net3[name].W.set_value(param_values[2*i])
#                  net3[name].b.set_value(param_values[2*i + 1])
#                  name = 'conv{}'.format(i+1)
#                  net4[name].W.set_value(param_values[2*i])
#                  net4[name].b.set_value(param_values[2*i + 1])
#              else:
#                  name = 'conv{}'.format(i+1)
#                  net[name].W.set_value(param_values[2*i])
#                  net[name].b.set_value(param_values[2*i + 1])
   
#==============================================================================
#==============================================================================
    
    def upscale(imageYCbCr,newH, newW):
        W,H = imageYCbCr.size
        seq = np.zeros((1,SEQ_LEN,newH,newW))
        #newH, newW = int(H*SCALE_BASE), int(W*SCALE_BASE)
        for k in xrange(SEQ_LEN):
            lowRes = imageYCbCr.resize((int(W*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1))),int(H*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1)))),resample=PIL.Image.LANCZOS)           
            #lowRes=imageYCbCr
            lowRes = np.array(lowRes.resize((newW,newH), resample=PIL.Image.LANCZOS)) * 1.0/255
            lowRes = lowRes[:,:,0]
            
            seq[0,k,:,:] = lowRes
        pred = probs(seq)*255
        pred = pred[0,0,0:newH,0:newW]
        pred = np.clip(pred,0,255)
        #scipy.misc.imsave(f+"_ycb.bmp",pred)
        #rint(pred.shape)
    


        bic = imageYCbCr.resize((newW,newH),resample=PIL.Image.LANCZOS)


        hightRes = np.array(bic)
        hightRes[:,:,0] = pred
        hightRes = hightRes.astype(np.uint8)
        ycbcr  = Image.fromarray(hightRes,'YCbCr')
        return ycbcr
   
    def test_bk(s):
        print("Enter test mode")
        onlyfiles = [f for f in listdir("/home/titanx1/Lasagne/examples/superRes/Test/Set5") if isfile(join("/home/titanx1/Lasagne/examples/superRes/Test/Set5", f))]
        avg_psnr = 0.0
        num_img = 0          
        for f in onlyfiles:
            input_file = join("/home/titanx1/Lasagne/examples/superRes/Test/Set5", f)
            originImg = Image.open(input_file)
            #originImg = originImg.resize((originImg.size[0]/2,originImg.size[1]/2),resample=PIL.Image.LANCZOS)
            
            ycbcr = originImg.convert('YCbCr') 
            W,H = ycbcr.size            
            lowRes = ycbcr.resize((int(W*1.0/s),int(H*1.0/s)), resample=PIL.Image.LANCZOS)
            
            highResYCbCr = upscale(lowRes, H, W)
            
            #compute psnr
            mse = np.sum(np.sum(((np.array(highResYCbCr)[:,:,0] - np.array(ycbcr)[:,:,0])**2)))/(H*W)
            #mse = np.sum(np.sum(((np.array(highResYCbCr)[:,:,0] - np.array(ycbcr)[:,:,0])**2)))/(H*W)
            psnr = 10*math.log10((255**2)/mse)
            print(" PSNR of {}: {}".format(f,psnr))
            #for loop in xrange(0):
            #    W,H = highResYCbCr.size
            #    highResYCbCr = upscale(highResYCbCr, int(H*SCALE_BASE), int(SCALE_BASE*W))
            avg_psnr = avg_psnr + psnr
            num_img = num_img + 1            
            rgb = highResYCbCr.convert('RGB')
            scipy.misc.imsave(f+"_x"+str(s)+"_sp2.bmp",rgb)
        print("(scale {})avreage PSNR {}".format(s,avg_psnr/num_img))  

    def test(s):
        print("Enter test mode")
       
        mat_contents = sio.loadmat('preprocessing_code/testingImg.mat')
        testData = mat_contents['testData']
        imgs = testData[0][0]
        print(imgs.shape)
        numImg,_,H,W,_ = imgs.shape
        avg_psnr = 0.0
        num_img = 0     
        input_data = imgs[:,:,:,:,0]*1.0/255
        highRes =  probs(input_data)*255
        
        
        for i in xrange(numImg):
            
           
#            seq = np.zeros((1,SEQ_LEN,H,W))
#            seq[0,:,:,:] = imgs[i,:,:,:]
#            
#            
#            seq= seq*1.0/255
#            highRes =  probs(seq)*255
#            highRes = np.round(highRes[0,0,:,:])
#            highRes = highRes.astype(np.uint8)
#           
#            #mse = np.sum(np.sum((highRes-gtImg)**2))/(H*W)
#            mse = np.sum(np.sum((highRes[2:H-2,2:W-2]-gtImg[2:H-2,2:W-2])**2))/((H-4)*(W-4))
#            
#            psnr = 10*math.log10((255**2)/mse)
#            print(" PSNR of {}: {}".format(i,psnr))
#           
#            avg_psnr = avg_psnr + psnr
#            num_img = num_img + 1     
            print(highRes.shape)
            scipy.misc.imsave("file_" + str(i)+"_x"+str(s)+"_sp.bmp",highRes[i,0,:,:])
            #hastableImg =np.zeros(highRes.shape)
            #print(hastableImg.shape)
        sio.savemat("output_sp.mat", {'Yimage':highRes})
        #print("(scale {})avreage PSNR {}".format(s,avg_psnr/num_img))
    if TEST_MODE:        
        
        
        #return 1
        onlyfiles = [f for f in listdir("/home/titanx1/Lasagne/examples/superRes/Test/Set5") if isfile(join("/home/titanx1/Lasagne/examples/superRes/Test/Set5", f))]
        avg_psnr = 0.0
        num_img = 0          
        for f in onlyfiles:
            input_file = join("/home/titanx1/Lasagne/examples/superRes/Test/Set5", f)
            originImg = Image.open(input_file)
            #originImg = originImg.resize((originImg.size[0]/2,originImg.size[1]/2),resample=PIL.Image.LANCZOS)
            print("==",originImg.size)
            ycbcr = originImg.convert('YCbCr') 
            W,H = ycbcr.size            
            lowRes = ycbcr.resize((int(W*1.0/3),int(H*1.0/3)), resample=PIL.Image.LANCZOS)
            
            highResYCbCr = upscale(lowRes, H, W)
            
            #compute psnr
            
            mse = np.sum(np.sum(((im2double(np.array(highResYCbCr)[:,:,0]) - im2double(np.array(ycbcr)[:,:,0]))**2)))/(H*W)
            psnr = 10*math.log10((1**2)/mse)
            #for loop in xrange(0):
            #    W,H = highResYCbCr.size
            #    highResYCbCr = upscale(highResYCbCr, int(H*SCALE_BASE), int(SCALE_BASE*W))
            avg_psnr = avg_psnr + psnr
            num_img = num_img + 1
            
            rgb = highResYCbCr.convert('RGB')
            scipy.misc.imsave(f+"_sp.bmp",rgb)
        print("avreage PSNR {}".format(avg_psnr/num_img))

    else:
       print("Training ...")
   
       
       for epoch in range(num_epochs):
           if(epoch % 2 == 0):
               test(2)
           #    test(3)
               #test(4)
           if (epoch % 20 == 0 and epoch > 0):
               lr.set_value(lr.get_value()*lr_decay)
               
           if (epoch % 5 == 0 and epoch > 0):
			print("saving ...")
			np.savez('MVDSR/MVDSR_'+str(epoch) + '.npz', *lasagne.layers.get_all_param_values(l_out))
           # In each epoch, we do a full pass over the training data:
  
           train_err = 0
           train_batches = 0
           start_time = time.time()
           for batch in iterate_minibatches(X_train, y_train, 64, shuffle=True):
               inputs, targets = batch
               #print(inputs.shape)         
               err = train(inputs, targets)
               train_err += err
               train_batches += 1
               if train_batches %10 == 0:
                   print("Batch cost: ",err)
               # Then we print the results for this epoch
           print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
           print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        
        
        
      
			
if __name__ == '__main__':
    main()

