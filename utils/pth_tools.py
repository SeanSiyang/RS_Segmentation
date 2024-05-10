import torch
import torch.nn as nn

#############################################################
# define model
model = nn.Sequential(
    nn.Linear(128, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# save parameters of model
torch.save(model.state_dict(), 'name_pt.pt')

# load parameters of model
loaded_model = nn.Sequential(
    nn.Linear(128, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loaded_model.load_state_dict(torch.load('name_pt.pt'))
# print the structure of the model
print(loaded_model)

#############################################################
# define the model
net = nn.Sequential(
    nn.Linear(128, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# save the complete model
torch.save(net, 'name_pt.pt')

# load the complete model, including the structure of the model and the parameters.
# The pt file includes the structure of the model, so we do not need to define the model before.
loaded_model = torch.load('name_pt.pt') 
print(loaded_model)

#############################################################
model = nn.Sequential(
    nn.Linear(128, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

input_sample = torch.randn(16, 128)
# save as the ONNX format
torch.onnx.export(model, input_sample, 'sample_model.onnx')

