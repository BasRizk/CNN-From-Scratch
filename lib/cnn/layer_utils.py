from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ########### DONE #############
                layer.grads[n] = v + lam * np.sign(layer.params[n])
                ########### END  #############

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ########### DONE #############
                layer.grads[n] = v + lam * 2 * layer.params[n]
                ########### END  #############


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # DONE: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size, in_height, in_width, _ = input_size
        output_shape = [
            batch_size,
            (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1,
            self.number_filters
        ]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # DONE: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        ############################################################################
        # print('kernels_dims', self.params[self.w_name].shape, '5,5,1,2')
        # print('img_shape', img.shape, '1,8,8,1', 'out_shape', output_shape, '1,4,4,2')
        
        img_wins = self.preprocess_img(
            img, kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )
        
        flatten_filters = np.vstack([
            self.params[self.w_name].T[i].T.reshape(
                self.kernel_size**2, self.input_channels
            ).T.reshape(self.kernel_size**2 * self.input_channels)
            for i in range(self.number_filters)
        ])
        
        # print('filters', self.params[self.w_name].shape)
        # print('img_wins', img_wins.shape, 'flatten_filters', flatten_filters.shape, flatten_filters[0].shape)
        output = np.vstack([
            img_wins@flatten_filters[i] + self.params[self.b_name][i]
            for i in range(self.number_filters)
        ])
        
        # print('sub raw output shape', (img_wins@flatten_filters[0]).shape)
        # print('raw output shape', output.shape)
        output = output.T.reshape(output_shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output
    
    def preprocess_img(self, img, kernel_size, padding, stride=None):
        padded_img = np.apply_along_axis(lambda x: np.pad(x, padding, mode='constant'), 1, img)
        padded_img = np.apply_along_axis(lambda x: np.pad(x, padding, mode='constant'), 2, padded_img)
        
        # print('padding', padding, padded_img.shape, img.shape)
        # print(padded_img)
        
        img_wins = np.lib.stride_tricks.sliding_window_view(
            padded_img, axis=(1,2),
            window_shape=(kernel_size, kernel_size)
        )

        # Note:
        # pushing all batches in one long array of shape 
        # (batch_size*num_wins x kernel_size*num_channels)
        # print('img_wins dims', img_wins.shape)
        img_wins = img_wins.reshape(
            *img_wins.shape[:-3],
            # img_wins.shape[-3],
            # np.prod(img_wins.shape[-2:]),
            np.prod(img_wins.shape[-3:]),
        )

        # print('img_wins dims before strides', img_wins.shape)
        # apply strides; skip rows
        if stride is not None:
            img_wins = img_wins[:, ::self.stride, ::self.stride]
        # print('img_wins dims after strides', img_wins.shape)
        img_wins = img_wins.reshape(
            np.prod(img_wins.shape[:-1]), 
            img_wins.shape[-1]
        )

        return img_wins
    
    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # print('output shape', self.get_output_size(img.shape))
        # print('w shape', self.params[self.w_name].shape, '4,4,3,12')
        # print('b shape', self.params[self.b_name].shape, '12,')
        # print('img shape', img.shape, '15,8,8,3')
        # print('dprev shape', dprev.shape, '15,4,4,12')
        
        
        img_wins = self.preprocess_img(
            img, kernel_size=self.kernel_size,
            padding=self.padding,
            stride=1
        )      
        # print('img_wins', img_wins.shape)
        # print(self.get_output_size(img))
        # print(self.kernel_size, dprev.T[0].T.shape)
        flatten_dprev = np.hstack([
            dprev.T[i].T.reshape(
                np.prod(dprev.shape[:-1]), 1
            )
            for i in range(self.number_filters)
        ])
        # print('flatten_dprev', flatten_dprev.shape)
        
        self.grads[self.w_name] = np.vstack(
            [
                flatten_dprev.T[i]@img_wins
                for i in range(self.number_filters)
            ]
        )
    
        # print('grad raw', self.grads[self.w_name].shape)
        self.grads[self.w_name] = np.transpose(
            self.grads[self.w_name].reshape(
                list(reversed(self.params[self.w_name].shape))
            ).T,
            axes=(1, 0, 2, 3)
        )
        # print('grad', self.grads[self.w_name].shape)

        self.grads[self.b_name] = np.sum(dprev, axis=(0,1,2))
        
        # rotated_filters = np.rot90(
        #     self.params[self.w_name],
        #     k=2,
        #     axes=(0,1)
        # )
        # print('rotated_filters', rotated_filters.shape)    
        
        # rotated_filters = self.preprocess_img(
        #     rotated_filters
        # ) 
        # flatten_weights = self.params[self.w_name].reshape(
        #     (self.kernel_size**2)*self.input_channels,
        #     self.number_filters,
        # )
        
        # flatten_filters = np.hstack([
        #     rotated_filters.T[i].T.reshape(
        #         self.kernel_size**2 * self.input_channels, 1
        #     )
        #     for i in range(self.number_filters)
        # ])
        # print('flatten_filters', flatten_filters.shape)
        # rotated_filters = rotated_filters.reshape(1, *rotated_filters.shape).T
        # one_rotated_filter =\
        #     rotated_filters.reshape(1, *rotated_filters.shape).T[0].T
        # print('one rotated filter', one_rotated_filter.shape)
        # print()
        # print('one rotated filter wins', 
        #     self.preprocess_img(
        #         one_rotated_filter,
        #         kernel_size=self.kernel_size,
        #         padding=self.kernel_size-1,
        #         stride=1
        #     ).shape)
        
        # print('img', img.shape)
        dimg = img.copy()
        # batch_size = 2 #dprev.shape[0]
        # for i in range(batch_size):
        #     dprev[i].T
        # rotated_filters_wins = np.vstack([
        #     np.expand_dims(self.preprocess_img(
        #         rotated_filters[i].T,
        #         kernel_size=dprev.shape[1],
        #         padding=self.kernel_size-1,
        #         stride=self.stride
        #     ), axis=0)
        #     for i in range(rotated_filters.shape[0])
        # ])
        
        # print('rotated_filters_wins', rotated_filters_wins.shape)
        
        # print((flatten_dprev.T[0]@rotated_filters_wins[0]).shape)
        # dimg = np.vstack(
        #     [
        #         flatten_dprev.T[i]@rotated_filters_wins[i]
        #         for i in range(self.number_filters)
        #     ]
        # ).reshape(img.shape)
        # print('rotated_filters_wins', rotated_filters_wins.shape)
        # dfeat = dprev@self.params[self.w_name].T
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # DONE: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        batch_size , input_height, input_width, num_channels = img.shape
        
        # print('img shape', img.shape)
        # print('pool_size', self.pool_size, 'stride', self.stride)
        img_wins = np.lib.stride_tricks.sliding_window_view(
            img, axis=(1,2), 
            window_shape=(self.pool_size, self.pool_size)
        )
        
        # print('img_wins dims before strides', img_wins.shape)
        # apply strides; skip rows
        img_wins = img_wins[:, ::self.stride, ::self.stride]
        
        # print('img_wins dims after strides', img_wins.shape) 
        
        # print(max_idx.shape)
        
        argmaxes = np.argmax(
            img_wins.reshape(
                *img_wins.shape[:-2], 
                self.pool_size**2
            ),
            axis=4
        )
        
        # print('argmaxes', argmaxes.shape)
        # print(argmaxes[0,0])
        self.unraveled_idx = np.array(list(zip(*np.unravel_index(
            argmaxes.flatten(),
            shape=(
                self.pool_size, self.pool_size
            )
        ))))
        
        # print('size of unravelled_idx', self.unraveled_idx.shape)
        
        # value_cnt = 0
        # maxes = np.zeros(self.unraveled_idx.shape[0])
        # for batch_i in range(img_wins.shape[0]):
        #     for h_i in range(img_wins.shape[1]):
        #         for w_i in range(img_wins.shape[2]):
        #             for ch_i in range(img_wins.shape[3]):
        #                 i, j = self.unraveled_idx[value_cnt]
        #                 maxes[value_cnt] = np.max(img_wins[batch_i,h_i,w_i,ch_i])
        #                 value_cnt += 1
        
        maxes = np.amax(img_wins, axis=(4,5))
        # print('maxes', maxes.shape)    

        output =\
            maxes.reshape(
                batch_size,
                (input_height - self.pool_size) // self.stride + 1,
                (input_width - self.pool_size) // self.stride + 1,
                num_channels        
            )
        
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # DONE: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        # print(img.shape)
        img_wins = np.lib.stride_tricks.sliding_window_view(
            dimg, axis=(1,2), 
            window_shape=(self.pool_size, self.pool_size),
            writeable=True
        )
        
        # print('img_wins dims before strides', img_wins.shape)
        # apply strides; skip rows
        img_wins = img_wins[:, ::self.stride, ::self.stride]
        # print('img_wins dims after strides', img_wins.shape)    
                
        # print('dprev shape', dprev.shape)
        dprev_flatten = dprev.flatten()
        value_cnt = 0
        for batch_i in range(img_wins.shape[0]):
            for h_i in range(img_wins.shape[1]):
                for w_i in range(img_wins.shape[2]):
                    for ch_i in range(img_wins.shape[3]):
                        i, j = self.unraveled_idx[value_cnt]
                        img_wins[batch_i,h_i,w_i,ch_i,i,j] += dprev_flatten[value_cnt]
                        value_cnt += 1
               
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
