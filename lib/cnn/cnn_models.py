from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## DONE: ##########
            ConvLayer2D(
                input_channels=3, 
                kernel_size=3, 
                number_filters=3
            ),
            MaxPoolingLayer(pool_size=2, stride=2, name='maxpool'),
            flatten(),
            fc(input_dim=27, output_dim=5, init_scale=0.02)
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # DATA SHAPE: (40000, 32, 32, 3)
            # 3072
            ConvLayer2D(
                input_channels=3, 
                kernel_size=3, 
                number_filters=9,
                stride=1,
                padding=1,
                init_scale=0.1,
                name='conv'
            ),
            MaxPoolingLayer(pool_size=2, stride=2, name='maxpool'),
            flatten(name="flat"),
            gelu(),
            
            fc(2304, 200, 0.1, name="fc1"),
            # dropout(keep_prob=0.8),
            gelu(name="gelu1"),
            
            fc(200, 100, 0.1, name="fc2"),
            gelu(name="gelu2"),
            
            # fc(50, 50, 0.1, name="fc3"),
            # gelu(name="gelu3"),
            
            # fc(100, 100, 0.1, name="fc4"),
            # gelu(name="gelu4"),
            
            # fc(100, 100, 0.1, name="fc5"),
            # gelu(name="gelu5"),
            
            fc(100, 20, 0.1, name="fc6")
            ########### END ###########
        )