"""
Main file of the repository with the main class definitions

@author: gallicch
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

class ReservoirCell(keras.layers.Layer):
#builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units, 
                 input_scaling = 1.0, bias_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units 
        self.state_size = units
        self.input_scaling = input_scaling 
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky #leaking rate
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #build the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value  = (self.spectral_radius / np.sqrt(self.units)) * (6/np.sqrt(12))
        W = tf.random.uniform(shape = (self.units, self.units), minval = -value,maxval = value)
        self.recurrent_kernel = W   
        
        #build the input weight matrix
        self.kernel = tf.random.uniform(shape = (input_shape[-1], self.units), minval = -self.input_scaling, maxval = self.input_scaling)
                         
        #initialize the bias 
        self.bias = tf.random.uniform(shape = (self.units,), minval = -self.bias_scaling, maxval = self.bias_scaling)
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]
    
    
class IRReservoirCell(keras.layers.Layer):
#builds the reservoir system for an input-routed ESN:
#the system contains a number of sub-reservoir systems, each fed by 1 specific input dimension
#inter-reservoir connections are modulated by specific hyper-parameters

    def __init__(self, units, 
                 input_scaling = 1.0, input_scaling2 = 0.0,
                 bias_scaling = 1.0,
                 inter_scaling = 1.0,
                 spectral_radius =0.99, 
                 leaky = 1, activation = tf.nn.tanh,
                 **kwargs):
        self.units = units 
        self.state_size = units
        self.input_scaling = input_scaling 
        self.inter_scaling = inter_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky #leaking rate
        self.activation = activation
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        
        #as for standard ESN, when building the recurrent weight matrix
        #uses circular law to determine the values of the recurrent weight matrix
        #rif. paper 
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. 
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        
        input_dimension = input_shape[-1] #this is the number of input units
        #which in this case is also the number of sub-reservoirs
        num_reservoirs = input_dimension
        num_units_reservoir = int(self.units / num_reservoirs)
        W = np.zeros(shape = (self.units, self.units))
        
        #recurrent kernel
        for i in range(num_reservoirs):
            value  = (self.spectral_radius[i] / np.sqrt(num_units_reservoir)) * (6/np.sqrt(12))
            W[num_units_reservoir * i :num_units_reservoir*(i+1),num_units_reservoir * i :num_units_reservoir*(i+1)] = tf.random.uniform(shape = (num_units_reservoir, num_units_reservoir), minval = -value,maxval = value)
            for j in range(num_reservoirs):
                if (i!=j):
                    W[num_units_reservoir * j :num_units_reservoir*(j+1),num_units_reservoir * i :num_units_reservoir*(i+1)] = tf.random.uniform(shape = (num_units_reservoir, num_units_reservoir), minval = -self.inter_scaling[i],maxval = self.inter_scaling[i])
        self.recurrent_kernel = W   

        #input weight matrix
        Win = np.zeros(shape = (input_dimension, self.units))
        for i in range(num_reservoirs):
            Win[i,num_units_reservoir*i:num_units_reservoir*(i+1)] = tf.random.uniform(shape = (1,num_units_reservoir), minval = -self.input_scaling[i], maxval = self.input_scaling[i])
        self.kernel = Win

        #bias 
        b = np.zeros(shape = (self.units,))
        for i in range(num_reservoirs):
            b[num_units_reservoir * i :num_units_reservoir*(i+1)] = tf.random.uniform(shape = (num_units_reservoir,), minval = -self.bias_scaling[i], maxval = self.bias_scaling[i])
        self.bias = b
        
        self.built = True


    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation!=None:
            output = prev_output * (1-self.leaky) + self.activation(input_part+ self.bias+ state_part) * self.leaky
        else:
            output = prev_output * (1-self.leaky) + (input_part+ self.bias+ state_part) * self.leaky
        
        return output, [output]    
    
class ESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units, output_units, output_activation,
                 input_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, 
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = ReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation = output_activation)
        ])   
        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        #the same is done on the validation set
        if 'validation_data' in kwargs:
            x_val,y_val = kwargs['validation_data']
            x_val_states = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_states, y_val)
        
        return self.readout.fit(x_train_states,y,**kwargs)
        
    def evaluate(self, x, y, **kwargs):
        x_train_states = self.reservoir(x)
        return self.readout.evaluate(x_train_states,y,**kwargs)
    
    
class IRESN(keras.Model):
    #Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    
    def __init__(self, units, output_units, output_activation,
                 input_scaling = 1., inter_scaling = 1., bias_scaling = 1.0, spectral_radius = 0.9,
                 leaky = 1, input_scaling2 = 1.,
                 activation = tf.nn.tanh,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.reservoir = keras.Sequential([
                    keras.layers.RNN(cell = IRReservoirCell(units = units,
                                                          input_scaling = input_scaling,
                                                          inter_scaling = inter_scaling,
                                                          bias_scaling = bias_scaling,
                                                          spectral_radius = spectral_radius,
                                                          leaky = leaky))
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation = output_activation)
        ])   
        
       
        
    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output
    
    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        x_train_states = self.reservoir(x)
        
        #the same is done on the validation set
        if 'validation_data' in kwargs:
            x_val,y_val = kwargs['validation_data']
            x_val_states = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_states, y_val)
        
        return self.readout.fit(x_train_states,y,**kwargs)
        
    def evaluate(self, x, y, **kwargs):
        x_train_states = self.reservoir(x)
        return self.readout.evaluate(x_train_states,y,**kwargs)
    
