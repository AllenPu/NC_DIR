import torch.nn as nn
import torchvision
import torch
from mlp import MLPFFNNeck


class ResNet_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_regression, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(
            args.model_depth))
        #
        output_dim = args.groups
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        mlp_output = 512
        #
        self.mlp = MLPFFNNeck(in_channels=fc_inputs, out_channels=mlp_output)
        #
        self.model_linear = nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #
        self.output_shape = mlp_output
        #self.mode = args.mode
        self.sigma = args.sigma

    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        print(f' input shape {z.shape}')
        z = self.mlp(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z
    
    def output_shape(self):
        return self.output_shape




class ResNet_regression_ddp(nn.Module):
    def __init__(self, args):
        super(ResNet_regression_ddp, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(
            args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear = nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma

    # g is the same shape of y
    def forward(self, x, g, mode='train'):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_predicted = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        y_chunk = torch.chunk(y_predicted, 2, dim=1)
        #
        g_hat, y_hat_all = y_chunk[0], y_chunk[1]
        #
        y_hat = torch.gather(y_hat_all, dim=1, index=g.to(torch.int64))
        #
        if mode == 'train':
            return g_hat, y_hat
        else:
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            y_gt = torch.gather(y_hat_all, dim=1, index=g.to(torch.int64))
            return g_index, y_hat, y_gt