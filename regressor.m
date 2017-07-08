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

function [beta] = regressor(H, Y, lamda)

ndata = size(H,1);
if ndata < size(H,2)
    beta = H'*pinv(H*H'+lamda*eye(ndata))*Y;
else
    beta = pinv(H'*H+lamda*eye(size(H,2)))*H'*Y;
end






