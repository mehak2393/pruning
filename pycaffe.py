#!/usr/bin/python

import caffe
import os
import pickle
import struct
import sys
import numpy as np
sys.dont_write_bytecode = True

if len(sys.argv) < 4:
    print "Usage: caffemodel.py <input_deploy_file> <input_model_file> <output_model_file>"
    sys.exit()
else:
    prototxt = sys.argv[1]
    orig = sys.argv[2] # original model (alex.caffemodel)
    comp = sys.argv[3] # compressed model (comp.caffemodel)
    
caffe.set_mode_gpu()
net = caffe.Net(prototxt, orig, caffe.TEST)
layers = filter(lambda x:'conv1' in x or 'fc' in x or 'ip' in x, net.params.keys())

# Extract boundary value at ratio x while sorting data
def read_boundary_value_with_ratio(data, ratio):
    print "pruning off "+str(ratio*100)+" % of this layer\n"
    arr = data
    arr = list(arr.reshape(arr.size))
    arr.sort(cmp=lambda x,y:cmp(abs(x), abs(y)))
    thresh = abs(arr[int(len(arr)*ratio)-1])
    return thresh

# Input: n-d dense array, Output: pruned array with threshold
def prune_dense(weight_arr, name="None", thresh=0.005, **kwargs):
    """Apply weight pruning with threshold """
    under_threshold = abs(weight_arr) < thresh
    weight_arr[under_threshold] = 0
    count = np.sum(under_threshold)
    return weight_arr

# How many percentages you want to apply pruning
ratio = {"conv1":0.91, "res2a_branch2a":0.91, "res2a_branch2b":0.75, "res2a_branch2c":0.75, "res2a_branch1" :0.75, "res2b_branch2a":0.75, "res2b_branch2b":0.75, "res2b_branch2c":0.75, "res2c_branch2a":0.75, "res2c_branch2b":0.75, "res2c_branch2c":0.75, "res3a_branch1" : 0.75,"res3b1_branch2a" : 0.75, "res3b1_branch2b" : 0.75, "res3b1_branch2c" : 0.75, "res3b2_branch2a" : 0.75, "res3b2_branch2b" : 0.75, "res3b2_branch2c" : 0.75, "res3b3_branch2a" : 0.75, "res3b3_branch2b" : 0.75, "res3b3_branch2c" : 0.75, "res4b1_branch2a" : 0.75, "res4b2_branch2a" : 0.75, "res4b3_branch2a" : 0.75,"res4b4_branch2a" : 0.75, "res4b5_branch2a" : 0.75, "res4b6_branch2a" : 0.75, "res4b7_branch2a" : 0.75 , "res4b8_branch2a" : 0.75, "res4b9_branch2a" : 0.75, "res4b10_branch2a" : 0.75 , "res4b11_branch2a":0.75 , "res4b12_branch2a":0.75 ,"res4b13_branch2a":0.75 ,"res4b14_branch2a":0.75 ,"res4b15_branch2a":0.75 ,"res4b16_branch2a":0.75 ,"res4b17_branch2a":0.75 ,"res4b18_branch2a":0.75 ,"res4b19_branch2a":0.75 ,"res4b20_branch2a":0.75 ,"res4b21_branch2a":0.75 ,"res4b22_branch2a":0.75 }

for idx, layer in enumerate(ratio):
    #print "layer name: ", i.name
    print "layer name: ", layer
    #print "width: ",      i.blobs[0].width
    #print "height: ",     i.blobs[0].height
    
    temp2 = net.params[layer][0].data
    temp = np.zeros(net.params[layer][0].shape, np.float32)
    np.copyto(temp, temp2)
    
    nnz_before = np.sum(temp != 0)
    
    boundary = read_boundary_value_with_ratio(temp, ratio[layer])
    temp = prune_dense(temp, name=layer, thresh=boundary)
    
    # re-write the compressed data on each network
    np.copyto(net.params[layer][0].data, temp)

    print "# of non-zero (before): ", nnz_before
    print "# of non-zero (after): ", np.sum(temp != 0)
    print ""

net.save(comp)
print " Compression is done! Output is dense model. "
