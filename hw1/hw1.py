"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)

        layer_list=[input_size]+hiddens+[output_size]
        layer_unit_num=zip(layer_list[:-1],layer_list[1:])
        self.linear_layers=[Linear(layer_pair[0],layer_pair[1],weight_init_fn,bias_init_fn) for layer_pair in layer_unit_num]
            
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(num_bn_layers)]

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
#         Complete the forward pass through your entire MLP.
#         self.input_collect=[x]
        for i in range(len(self.linear_layers)):
            x=self.linear_layers[i].forward(x)
            if self.bn and i<self.num_bn_layers:#exist bachnorm,start from layer 0
                if self.train_mode == True:
                    x=self.bn_layers[i].forward(x)
                else:
                    x=self.bn_layers[i].forward(x,eval=True)
            x=self.activations[i](x)
#             self.input_collect.append(x)
            self.output=x
        return x

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for item in self.linear_layers:
            item.dW.fill(0.0)
            item.db.fill(0.0)
        if self.bn:
            for item in self.bn_layers:
                item.dgamma.fill(0.0)
                item.dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].momentum_W=self.linear_layers[i].momentum_W*self.momentum-self.lr*self.linear_layers[i].dW
            self.linear_layers[i].momentum_b=self.linear_layers[i].momentum_b*self.momentum-self.lr*self.linear_layers[i].db
            self.linear_layers[i].W+=self.linear_layers[i].momentum_W
            self.linear_layers[i].b+=self.linear_layers[i].momentum_b
            
        # Do the same for batchnorm layers
        if self.bn:
            for item in self.bn_layers:
                item.gamma-=self.lr*item.dgamma
                item.beta-=self.lr*item.dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        
        #softmax layer
        criterion_forward=self.criterion.forward(self.output,labels)
        derivative=self.criterion.derivative() #output:batchsize,10
        #hidden layers
        for i in range(len(self.linear_layers)-1,-1,-1):
            #input-hidden1,hidden1-2,...hidden-1-z,then use softmax on z,num of linear layers=num of hiddens +1
            #range(4,-1,-1):4,3,2,1,0, so can represent hidden layer+input layer
            derivative *= self.activations[i].derivative()
            if i+1<=self.num_bn_layers:
                derivative =self.bn_layers[i].backward(derivative)
            derivative=self.linear_layers[i].backward(derivative) 
        
        return derivative

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val
    idxs = np.arange(len(trainx))
#     print(len(idxs))
    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...#shuffle data
    for e in range(nepochs):#for each epoch

        # Per epoch setup ...
        np.random.shuffle(idxs)
        mlp.train()#train_mode=true
        epoch_loss=[]
        epoch_error=[]
        for b in range(int(len(trainx)/batch_size)):#for each batch
            mlp.zero_grads()
            trainx_batch=trainx[idxs][b*batch_size:(b+1)*batch_size]
            trainy_batch=trainy[idxs][b*batch_size:(b+1)*batch_size]
#             print('batch-shape:',trainx_batch.shape)
            mlp.forward(trainx_batch)
            epoch_loss.append(mlp.total_loss(trainy_batch)/batch_size)
#             print('loss of batch:',epoch_loss[-1])
            epoch_error.append(mlp.error(trainy_batch)/batch_size)
            mlp.backward(trainy_batch)
            mlp.step()
            
        training_losses[e]=np.array(epoch_loss).mean()
        training_errors[e]=np.array(epoch_error).mean()
        print('epoch:',e+1)
        print('training_losses:',training_losses[e])
        print('training_errors:',training_errors[e])
        
        #validation
        mlp.eval()#train_mode=false
        epoch_loss=[]
        epoch_error=[]
        for b in range(int(len(valx)/batch_size)):
            valx_batch=valx[b*batch_size:(b+1)*batch_size]
            valy_batch=valy[b*batch_size:(b+1)*batch_size]
            mlp.forward(valx_batch)
            epoch_loss.append(mlp.total_loss(valy_batch)/batch_size)
            epoch_error.append(mlp.error(valy_batch)/batch_size)
                   
        validation_losses[e]=np.array(epoch_loss).mean()
        validation_errors[e]=np.array(epoch_error).mean()
        print('validation_losses:',validation_losses[e])
        print('validation_errors:',validation_errors[e])
        # Accumulate data...
    
    # Cleanup ...
    # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)

