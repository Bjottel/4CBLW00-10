#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
MODEL_PATH = 'model_full.pth'

# Copy paste from Model.ipynb
class NeuralNetwork(nn.Module):
    def __init__(self, conv_config, fc_config, activation):
        super().__init__()
        
        activation_func = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
        }[activation]
        

        conv_layers = []
        last_out_channels = 1
        conv_out_len = INPUT_LENGTH
        for config in conv_config:
            conv_layers.append(nn.Conv1d(in_channels=last_out_channels, out_channels=config['conv_outchannels'], kernel_size=config['conv_kernelsize']))
            conv_layers.append(nn.BatchNorm1d(config["conv_outchannels"]))
            conv_layers.append(activation_func)
            
            if (config['pooling'] == 'max'):
                conv_layers.append(nn.MaxPool1d(config['pooling_kernelsize']))
            elif (config['pooling'] == 'avg'):
                conv_layers.append(nn.AvgPool1d(config['pooling_kernelsize']))
            last_out_channels = config['conv_outchannels']
            conv_out_len = (conv_out_len - (config['conv_kernelsize'] - 1)) // config['pooling_kernelsize']
            
            # If the kernel sizes get a bit too big the one of the dimensions of the out-tensor can become 0 (or even negative)
            # This is invalid so we just return an error if this happens and try again with a different configuration
            if conv_out_len <= 0:
                raise ValueError(f"Output length became zero or negative after layer with config: {config}")
        self.conv_stack = nn.Sequential(*conv_layers)

        fc_layers = []
        last_out = conv_out_len * last_out_channels
        
        if last_out <= 0:
            raise ValueError(f"Output length became zero or negative")
            
        fc_layers.append(nn.Flatten())
        for config in fc_config:
            fc_layers.append(nn.Linear(last_out, config['fc_size']))
            fc_layers.append(activation_func)
            fc_layers.append(nn.Dropout(config['dropout']))
            last_out = config['fc_size']
        fc_layers.append(nn.Linear(last_out, num_classes))
        self.fc_stack = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

# Model was saved on cuda so should be brought back to cpu
model = torch.load(MODEL_PATH).to('cpu')
model.eval()


# In[14]:


def predict(inputs, use_optimized_thresholds=False):
    thresholds = [0.5, 0.5, 0.5]
    if use_optimized_thresholds:
        thresholds = [0.6900000000000001, 0.09, 0.23]
    thresholds_tensor = torch.tensor(thresholds).unsqueeze(0)
        
    input_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs)
        
        predicted_labels = (probabilities > thresholds_tensor).squeeze(0).bool()
        
    return predicted_labels.tolist(), probabilities.squeeze(0).tolist()

