import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import jcamp  #if it doesn't work: put pip install jcamp in command prompt
import torch
import torch.nn as nn
#MODEL_PATH = 'model_full.pth'
MODEL_PATH="C:/Users/Pleun/Documents/Data science CBL/model_full.pth"
# Copy paste from Model.ipynb

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import json
MIN_WAVENUMBER=400
MAX_WAVENUMBER=4000
def is_jdx_valid(jdx_file):
    # If a jdx file can't be read by the jcamp library we just consider it invalid
    try:
        jcamp_dict = jcamp.jcamp_readfile(jdx_file)
    except Exception as e:
        error_reasons[0] = error_reasons[0] + 1
        return False, {}
    
    # Some jcamp files contain a full spectrum and peak tables/other data. In these cases the jcamp_dict['children'] exists.
    # We only care about the full spectrum so we ignore the peak tables/other data. The full spectrum is always the first child.
    if ('children' in jcamp_dict):
        jcamp_dict = jcamp_dict['children'][0]
    
    # Some very old spectra don't have jdx files yet NIST provides downloads for them anyway.
    # These files don't have any measurements and sometimes don't even include the npoints tag so we filter them out.
    if ((not 'npoints' in jcamp_dict) or jcamp_dict['npoints'] < 1):
        error_reasons[1] = error_reasons[1] + 1
        return False, jcamp_dict
    
    # Some spectra use y units like absorption/refraction index as their y units.
    # These can't be easily converted to absorbance/transmittance and the jcamp library parses them wrong so we filter them out.
    if (jcamp_dict['yunits'].lower() != 'transmittance' and jcamp_dict['yunits'].lower() != 'absorbance'):
        error_reasons[2] = error_reasons[2] + 1
        return False, jcamp_dict
    
    if (not 'xunits' in jcamp_dict):
        error_reasons[3] = error_reasons[3] + 1
        return False, jcamp_dict
    
    # We first have to make sure our first/last x are given in cm^-1
    if ('minx' in jcamp_dict):
        min_x = jcamp_dict['minx']
    else:
        min_x = min(jcamp_dict['x'])
        
    if ('maxx' in jcamp_dict):
        max_x = jcamp_dict['maxx']
    else:
        max_x = max(jcamp_dict['x'])

    if (jcamp_dict['xunits'].lower() == 'micrometers'):
        min_x = 1e4 / min_x
        max_x = 1e4 / max_x
        
    # Since we only care for the spectrum from wavenumbers MIN_WAVENUMBER to MAX_WAVENUMBER, we disregard any coverage outside that range.
    min_x = max(min_x, MIN_WAVENUMBER)
    max_x = min(max_x, MAX_WAVENUMBER)
    
    # Some samples only cover a very small range of wavelengths. This is generally bad so we filter them out.
    cover = (max_x - min_x) / (MAX_WAVENUMBER - MIN_WAVENUMBER)
    if (cover < COVER_THRESHOLD):
        error_reasons[4] = error_reasons[4] + 1
        return False, jcamp_dict
    
    return True, jcamp_dict
def preprocess_jdx(jcamp_dict, baseline_correct=True):  
    yvals_raw = jcamp_dict['y']
    yunits = jcamp_dict['yunits'].lower()
    
    xvals_raw = jcamp_dict['x']
    xunits = jcamp_dict['xunits'].lower()
    
    if (xunits == 'micrometers'):
        # wavenumber = 10^4 / wavelength
        xvals_raw = 1e4 / xvals_raw 
        
    # Convert all y values to absorbance
    if (yunits == 'transmittance'):
        yvals_raw = -np.log10(np.clip(yvals_raw, 1e-8, None))
        
    if baseline_correct:
        yvals_raw = yvals_raw - np.min(yvals_raw)
    
    # We fill missing values with zeroes
    interpolator = interp1d(xvals_raw, yvals_raw, kind='linear', fill_value=0, bounds_error=False)
    xvals_new = np.linspace(MIN_WAVENUMBER, MAX_WAVENUMBER, MAX_WAVENUMBER - MIN_WAVENUMBER)
    yvals_new = interpolator(xvals_new)
    
    max_y = np.max(yvals_new)
    if max_y > 0:
        yvals_new = yvals_new / max_y
      
    #plt.plot(xvals_new, yvals_new, '.', label='Interpolated data')
    #plt.plot(xvals_raw, yvals_raw, '.', label='Raw data')
    #plt.legend()
    #plt.show()
    
    return yvals_new
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
model = torch.load(MODEL_PATH,weights_only=False,map_location=torch.device('cpu'))
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

def open_jcamp_file():
    path = filedialog.askopenfilename(
        title="Open JCAMP File"#, filetypes=[("JCAMP files", "*.jdx")]
    )
    if path:
        display_answer_data(path)

def display_answer_data(path):
    try:
        data = jcamp.jcamp_readfile(path)
        
        predicted_labels, probabilities = predict(preprocess_jdx(data))

        scores = [round(val * 100, 2) for val in probabilities]
        groups = ["Phenol", "Aldehyde", "Benzene Ring","Toxicity"]
        if  scores[1]>30 and scores[2]>30 or scores[0]>30:
            scores.append("Toxic")
        elif scores[2]>30:
            scores.append("Possibly Toxic, but unclear")

        elif scores[1]>30:
            scores.append("Toxicity risk")

        tree.delete(*tree.get_children())
        tree["columns"] = ("Functional Group", "Confidence (%)")
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for g, sc in zip(groups, scores):
            tree.insert("", "end", values=(g, sc))

        status_label.config(text=f"JCAMP file loaded: {path}")

    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("JCAMP File Viewer")

open_button = tk.Button(root, text="Open JCAMP File", command=open_jcamp_file)
open_button.pack(padx=20, pady=10)

tree = ttk.Treeview(root, show="headings")
tree.pack(padx=20, pady=20, fill="both", expand=True)

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.pack()

root.mainloop()
