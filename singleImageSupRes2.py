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



from PIL import Image

import sys
from os import listdir
from os.path import isfile, join

sys.path.insert(0,'/home/titanx1/Downloads/caffe-master/python')
import caffe

net_caffe = caffe.Net('VDSR_net.prototxt', '_iter_VDSR_Official.caffemodel', caffe.TEST)
layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

# set test mode
TEST_MODE = False

SCALE_BASE = 1.05

SCALE_FACTOR = 2

SEQ_LEN = 1

# Optimization learning rate
LEARNING_RATE = 0.0001#0.00001

# All gradients above this will be clipped
GRAD_CLIP = 0.01

# Number of epochs to train the net
NUM_EPOCHS = 5000

PATCH_SIZE = 33

BATCH_SIZE = 30

STRIDE = 14


#Lasagne Seed for Reproducibility
#lasagne.random.set_rng(np.random.RandomState(1))


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
    with h5py.File('/home/titanx1/Downloads/caffe-master/examples/VDSR/train.h5','r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
    return data,label
    	
def main(num_epochs = NUM_EPOCHS):

	 # Load the dataset
    print("Loading data...")
    #X_train, y_train =  load_dataset("91")
    X_train, y_train =  load_hdf5_data()
#==============================================================================
#     print(X_train.shape)
#     N= X_train.shape[0]
#     idx = np.random.randint(1,N,10)#
#     for i in idx:
#         scipy.misc.imsave("high_"+ str(i)+"_" + str(0) + ".png",y_train[i,0,:,:])
#         for j in xrange(1):
#     			scipy.misc.imsave("low_"+ str(i)+"_" + str(j) + ".png",X_train[i,j,:,:])
#==============================================================================
	# create Theano variables for input and target minibatch
	
    input_values = T.tensor4('X')
	#target_var = T.tensor4('Y')

	# construct CNN net
    net = {}
	#net['input'] = lasagne.layers.InputLayer((None, 1, PATCH_SIZE, PATCH_SIZE),input_values)
    if TEST_MODE:
        net['input'] = lasagne.layers.InputLayer((None, SEQ_LEN, None, None),input_values)
    else:
        net['input'] = lasagne.layers.InputLayer((None, SEQ_LEN, PATCH_SIZE, PATCH_SIZE),input_values)	
	
    net['slice1'] = lasagne.layers.SliceLayer(net['input'], indices=slice(0, 1), axis=1)
    net['slice2'] = lasagne.layers.SliceLayer(net['input'], indices=slice(1, 2), axis=1)
    net['slice3'] = lasagne.layers.SliceLayer(net['input'], indices=slice(2, 3), axis=1)
    net['slice4'] = lasagne.layers.SliceLayer(net['input'], indices=slice(3, 4), axis=1)
    

    for i in xrange(SEQ_LEN):
        namelayer = 'conv1_{}'.format(i+1)
        name_slice_input = 'slice{}'.format(i+1)
        #net[namelayer]  = lasagne.layers.Conv2DLayer(net[name_slice_input],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
        net[namelayer]  = lasagne.layers.Conv2DLayer(net[name_slice_input],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify)
        net[namelayer].W.tag.grad_scale = 1
        net[namelayer].b.tag.grad_scale = 0.1

    #net['conv1_2']  = lasagne.layers.Conv2DLayer(net['slice2'],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
    #net['conv1_2'].W.tag.grad_scale = 1
    #net['conv1_2'].b.tag.grad_scale = 0.1

    
    
    for i in xrange(1,19):
        for j in xrange(SEQ_LEN):
            namelayer ='conv{}_{}'.format(i+1,j+1)
            prvlayername = 'conv{}_{}'.format(i,j+1)
            net[namelayer] =  lasagne.layers.Conv2DLayer(net[prvlayername],64,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.rectify,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
            if i<30:
                net[namelayer].W.tag.grad_scale = 1
                net[namelayer].b.tag.grad_scale = 0.1
            else:
                net[namelayer].W.tag.grad_scale = 1
                net[namelayer].b.tag.grad_scale = 0.1
    
    for i in xrange(SEQ_LEN):
        namelayer = 'conv20_{}'.format(i+1)
        print(namelayer)
        preNameLayer = 'conv19_{}'.format(i+1)
        net[namelayer] = lasagne.layers.Conv2DLayer(net[preNameLayer],1,(3,3),pad = 1, nonlinearity =  lasagne.nonlinearities.linear,W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))

    net['sum1']= lasagne.layers.ElemwiseSumLayer({net['conv20_1'], net['slice1']})
#==============================================================================
#     net['sum2']= lasagne.layers.ElemwiseSumLayer({net['conv20_2'], net['slice2']})
#     net['sum3']= lasagne.layers.ElemwiseSumLayer({net['conv20_3'], net['slice3']})
#     net['sum4']= lasagne.layers.ElemwiseSumLayer({net['conv20_4'], net['slice4']})
#     
#     net['concat'] = lasagne.layers.ConcatLayer((net['sum1'],net['conv3_2'],net['sum2'],net['sum3'],net['sum4']), axis = 1) 
#     net['conv21'] = lasagne.layers.Conv2DLayer(net['concat'],1,(3,3),pad = 1, W=lasagne.init.GlorotNormal(gain=math.sqrt(2)),b=lasagne.init.Constant(0.))
#==============================================================================

    print("Building network SRCNN...")
    #l_out2 = lasagne.layers.ReshapeLayer(net['conv3'],(BATCH_SIZE*SEQ_LEN,1,c,PATCH_SIZE))
    l_out = net['sum1']
    # Theano tensor for the targets
    target_values = T.tensor4('target_output')
    
    network_output = lasagne.layers.get_output(l_out)
    network_output2 = lasagne.layers.get_output(net['conv20_1'])
    #loss = 0.5*PATCH_SIZE**2*lasagne.objectives.squared_error(network_output, target_values).mean()
    loss = 0.5*PATCH_SIZE**2*squared_error(network_output, target_values).mean()
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
    lr_decay = np.array(0.5,dtype=theano.config.floatX)
    updates = lasagne.updates.momentum(grads, params, learning_rate=lr,momentum=0.9)
    #updates = lasagne.updates.adagrad(grads, params, learning_rate=MYLEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([input_values, target_values], loss, updates = updates, allow_input_downcast = True)
    probs = theano.function([input_values], network_output, allow_input_downcast = True)
    
    probs2 = theano.function([input_values], network_output2, allow_input_downcast = True)
    #compute_cost = theano.function([input_values, target_values], loss, allow_input_downcast = True)
    #DATA_SIZE = X_train[0]
    
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#    print(layers_caffe['conv1'].blobs[1].data)    
    for item in net.items():
            name,layer =item 
            if(string.find(name,'conv') >=0):
                print(name[0:-2])
                print(layer)
                W = layers_caffe[name[0:-2]].blobs[0].data[:,:,::-1,::-1]
                b = layers_caffe[name[0:-2]].blobs[1].data
                layer.W.set_value(np.transpose(W,(0,1,3,2)))
                layer.b.set_value(b)
                #layer.W.set_value(layers_caffe[name[0:-2]].blobs[0].data[...,::-1,::-1])                
                #layer.b.set_value(layers_caffe[name[0:-2]].blobs[1].data)
                
#==============================================================================
           
        
    #with np.load('MSRCNN/Test_10.npz') as fi:
    #      param_values = [fi['arr_%d' % i] for i in range(len(fi.files))]	
    #      lasagne.layers.set_all_param_values(l_out, param_values)
    #with np.load('MSRCNN/VDSR_20.npz') as fi:
    #      param_values = [fi['arr_%d' % i] for i in range(len(fi.files))]	
    ##      for i in xrange(len(param_values) // 2):
    #          name = 'conv{}_1'.format(i+1)
    #          net[name].W.set_value(param_values[2*i])
    #          net[name].b.set_value(param_values[2*i + 1])
#==============================================================================
#               name = 'conv{}_2'.format(i+1)
#               net[name].W.set_value(param_values[2*i])
#               net[name].b.set_value(param_values[2*i + 1])
#               name = 'conv{}_3'.format(i+1)
#               net[name].W.set_value(param_values[2*i])
#               net[name].b.set_value(param_values[2*i + 1])
#               name = 'conv{}_4'.format(i+1)
#               net[name].W.set_value(param_values[2*i])
#               net[name].b.set_value(param_values[2*i + 1])
#==============================================================================
   
#==============================================================================
#==============================================================================
    
    def upscale(imageYCbCr,newH, newW):
        W,H = imageYCbCr.size
        seq = np.zeros((1,SEQ_LEN,newH,newW))
        #newH, newW = int(H*SCALE_BASE), int(W*SCALE_BASE)
        for k in xrange(SEQ_LEN):
            lowRes = imageYCbCr.resize((int(W*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1))),int(H*1.0/(SCALE_BASE**(SEQ_LEN -  k - 1)))),resample=PIL.Image.BICUBIC)           
            #lowRes=imageYCbCr
            lowRes = np.array(lowRes.resize((newW,newH), resample=PIL.Image.BICUBIC)) * 1.0/255
            lowRes = lowRes[:,:,0]
            #container = np.zeros((512,512))
           # container[0:newH,0:newW] = lowRes
            seq[0,k,:,:] = lowRes
        pred = probs(seq)*255
        pred = pred[0,0,0:newH,0:newW]
        pred = np.clip(pred,0,255)
        #scipy.misc.imsave(f+"_ycb.bmp",pred)
        #rint(pred.shape)
    


        bic = imageYCbCr.resize((newW,newH),resample=PIL.Image.BICUBIC)


        hightRes = np.array(bic)
        hightRes[:,:,0] = pred
        hightRes = hightRes.astype(np.uint8)
        ycbcr  = Image.fromarray(hightRes,'YCbCr')
        return ycbcr
   
    def test(s):
        print("Enter test mode")
       
        mat_contents = sio.loadmat('testingImgSet14Scale'+str(s)+'.mat')
        imgs = mat_contents['testingImg']
        _,numImg = imgs.shape
        numImg = numImg -2
        avg_psnr = 0.0
        num_img = 0          
        for i in xrange(numImg):
            lowRes = imgs[0,2+i][:,:,0]
            gtImg =  imgs[0,2+i][:,:,1]
            H,W = gtImg.shape
            seq = np.zeros((1,1,H,W))
            seq[0,0,:,:] = lowRes*1.0/255
            highRes =  probs(seq)*255
            highRes = highRes[0,0,:,:]
            #highRes = np.clip(highRes,0,255)
            
            
            #compute psnr
            #mse = np.sum(np.sum(((np.array(highResYCbCr)[:,:,0] - np.array(ycbcr)[:,:,0])**2)))/(H*W)
            mse = np.sum(np.sum((highRes-gtImg)**2))/(H*W)
            psnr = 10*math.log10((255**2)/mse)
            print(" PSNR of {}: {}".format(i,psnr))
            #for loop in xrange(0):
            #    W,H = highResYCbCr.size
            #    highResYCbCr = upscale(highResYCbCr, int(H*SCALE_BASE), int(SCALE_BASE*W))
            avg_psnr = avg_psnr + psnr
            num_img = num_img + 1            
            scipy.misc.imsave("file " + str(i)+"_x"+str(s)+"_sp.bmp",highRes)
        print("(scale {})avreage PSNR {}".format(s,avg_psnr/num_img))   
        
                
    if TEST_MODE:
    	print("testing ...")
        test(SCALE_FACTOR)
        return 1
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
            lowRes = ycbcr.resize((int(W*1.0/SCALE_FACTOR),int(H*1.0/SCALE_FACTOR)), resample=PIL.Image.BICUBIC)
            
            highResYCbCr = upscale(lowRes, H, W)
            
            #compute psnr
            mse = np.sum(np.sum(((np.array(highResYCbCr)[:,:,0] - np.array(ycbcr)[:,:,0])**2)))/(H*W)
            #im1,im2 = im2double(np.array(highResYCbCr)), im2double(np.array(ycbcr))
            #mse = np.sum(np.sum(((im1 - im2)**2))) / (H*W)
            psnr = 10*math.log10((255**2)/mse)
            print("PSNR of {} (x{}) : {}".format(f,SCALE_FACTOR,psnr))
            #for loop in xrange(0):
            #    W,H = highResYCbCr.size
            #    highResYCbCr = upscale(highResYCbCr, int(H*SCALE_BASE), int(SCALE_BASE*W))
            avg_psnr = avg_psnr + psnr
            num_img = num_img + 1
            
            rgb = highResYCbCr.convert('RGB')
            scipy.misc.imsave(f+"_x" +str(SCALE_FACTOR)+ "_sp.bmp",rgb)
        print("avreage PSNR {}".format(avg_psnr/num_img))

    else:
       print("Training ...")
   
       
       for epoch in range(num_epochs):
           if ( epoch % 5 ==0):
               test(2)
               test(3)
               test(4)
           if (epoch % 10 == 0 and epoch > 0):
               lr.set_value(lr.get_value()*lr_decay)
               
           if (epoch % 5 == 0 and epoch > 0):
               print("saving ...")
               np.savez('MSRCNN/Test_'+str(epoch) + '.npz', *lasagne.layers.get_all_param_values(l_out))
           # In each epoch, we do a full pass over the training data:
  
           train_err = 0
           train_batches = 0
           start_time = time.time()
           for batch in iterate_minibatches(X_train, y_train, 96, shuffle=True):
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

