
% Outlier-robust extreme learning machine, Version 2.0
% This is an implementation of the algorithm for "SinC" function regression
%
% Please cite the following paper if you use this code:
%
% Zhang, Kai, and Minxia Luo. "Outlier-robust extreme learning machine for regression problems."
% Neurocomputing 151 (2015): 1519-1527.
% =========================================================================
%   written by Kai Zhang. Email: zhkmath@163.com 
%   Website: https://sites.google.com/site/cskaizhang/home
% =========================================================================

function nn = elm_initialization(nn)

% biases and input weights

nn.b = 2*rand(nn.hiddensize,1)-1;
nn.W = 2*rand(nn.hiddensize, nn.inputsize)-1;
nn.W1 = 2*rand(nn.hiddensize, nn.inputsize)-1;
nn.b1 = 2*rand(nn.hiddensize,1)-1;

% nn.b = randn(nn.hiddensize,1);
% nn.W = randn(nn.hiddensize, nn.inputsize);


if nn.orthogonal
    if nn.hiddensize > nn.inputsize
        nn.W = orth(nn.W);
        nn.W1 = orth(nn.W1);
    else
        nn.W = orth(nn.W')';
        nn.W1 = orth(nn.W1')';
    end
    nn.b=orth(nn.b);
    nn.b1=orth(nn.b1);
end

end

