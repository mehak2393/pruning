import numpy as np
import sys
import caffe
#gpu_id = sys.args[1]
#caffe.set_mode_gpu()
#caffe.set_device(gpu_id)
MODEL_FILE = 'models/lenet5.prototxt'
PRETRAINED = 'models/caffe_lenet5_original.caffemodel'
net = caffe.Net('lenet5.prototxt', 'caffe_lenet5_original.caffemodel',caffe.TEST)
print "\nnet.inputs =", net.inputs ## names of layers
print "\ndir(net.blobs) =", dir(net.blobs) ## data in the layers
print "\ndir(net.params) =", dir(net.params) ## for weight and biases in the netwrk
print "\nconv shape = ", net.blobs['conv'].data.shape
w1 = net.params[‘conv’][0]
b1 = net.params[‘conv’][1]
net.forward()
net.save('lenet5_compressed.caffemodel') #saving the model

###pruning
threshold = 0.0001
for i in range(0,3):
	

