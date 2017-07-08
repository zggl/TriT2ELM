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

function [w] = weight_fun(e, wfun, s,alg_ind)

if nargin < 2
    wfun = '1';
    s    = median(abs(e(:)))/0.6745;
    % default setting: s = median(abs(e))/0.6745;
    % another equivalent setting when the mean of e is zero: s = iqr(e)/(2*0.6745);
end

w  = zeros(size(e));

switch(wfun)
    
    case{'1','default'}
        
        % default setting: s = iqr(e)/(2*0.6745);
        % another equivalent setting when the mean of e is zero: s = iqr(e)/(2*0.6745);
        a=2.5;  b=3;
        ind1    = (abs(e/s)<=a);
        w(ind1) = 1;
        ind2    = ((a<abs(e/s)) & (abs(e/s)<b));
        w(ind2) = (b-abs(e(ind2)/s))/(b-a);
        ind3    = ~(ind1|ind2);
        w(ind3) = 1e-4;
        
    case{'2','bisquare','b'}
        tune = 4.685;
        r    = e/(tune*s);
        w    = (abs(r)<1) .* (1 - r.^2).^2;
        
    case{'3','huber','h'}
        tune = 1.345;
        r    = e/(tune*s);
        w    = 1./max(1,abs(r));
        
    case{'4','lp','l'}
        p = 1; % L1-norm
        w = 1./max(0.0001,abs(e).^(2-p));
       % w = w/sum(w);
        %     case{'5','andrews'}
        %
        %         tune  = 1.339;
        %         r     = e/(tune*s);
        %         w     = (abs(r)<pi) .* sin(r) ./ r;
        %
        %     case{'6','fair'}
        %         tune = 1.400;
        %         r    = e/(tune*s);
        %         w    = 1 ./ (1 + abs(r));
        %
        %     case{'7','cauchy'}
        %         tune = 2.385;
        %         r    = e/(tune*s);
        %         w    = 1 ./ (1 + r.^2);
        %
        %     case{'8','logistic'}
        %         tune = 1.205;
        %         r    = e/(tune*s);
        %         w    = tanh(r) ./ r;
        %
        %     case{'9','talwar'}
        %         tune = 2.795;
        %         r    = e/(tune*s);
        %         w    = 1 * (abs(r)<1);
        %
        %     case{'10','welsch'}
        %         tune = 2.985;
        %         r    = e/(tune*s);
        %         w    = exp(-(r.^2));
    case{'5'} 
        w=1
        
end














