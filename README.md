Extract the 4096 dimension activations from the VGGNet built in Keras. The last two layer have been cut off - The last Softmax Layer and Dropout layer.

This has the following dependencies:

1) Keras
2) Theano
3) OpenCV
4) Numpy

To run the code : 

sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kerasvgg.py 

Link for Pre-Trained Weights : https://www.dropbox.com/s/bvn05hji37lhfoa/vgg16_weights.h5?dl=0
