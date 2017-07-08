
% =========================================================================
% Outlier-robust extreme learning machine, Version 2.0
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for "SinC" function regression
%
% Please cite the following paper if you use this code:
%
% Zhang, Kai, and Minxia Luo. "Outlier-robust extreme learning machine for regression problems."
% Neurocomputing 151 (2015): 1519-1527.
%
%--------------------------------------------------------------------------

function [nn, acc_train] = elm_train(X, Y, nn, alg_ind)

% beta f(Wx+b) = y

tic;

ndata = size(X,2);
tempH = nn.W*X + repmat(nn.b,1,ndata);
tempH2        = nn.W1*X + repmat(nn.b,1,ndata);
[N,L]=size(tempH);
switch lower(nn.activefunction{alg_ind})
    case{'s','sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case{'t','tanh'}
        H = tanh(tempH);
    case{'r','rbf'} 
        tempH=-((bsxfun(@minus,X, nn.W)).^2);   
        H=exp(tempH./15); 
%         y1=exp(tempH)+0.0001;
    case{'t2s','t2sig','t2sigmoid'}
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr        = nn.W*(X-1 )+ repmat(nn.b,1,ndata);
        tempHl        = nn.W*(X-2 )+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        H=(Hu+Hpr+Hl)/3;  
    case{'t2srnd','t2sigrnd','t2sigmoidrnd'}
        nn.sigran_mean=[0.5+0.5*rand,0.5*rand+1.6];
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr        = nn.W*(X-nn.sigran_mean(1))+ repmat(nn.b,1,ndata);
        tempHl        = nn.W*(X- nn.sigran_mean(2))+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        H=(Hu+Hpr+Hl)/3;
    case{'t2s2','t2sig2','t2sigmoid2'}
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr  = nn.W*(X-1 )+ repmat(nn.b,1,ndata);
        tempHl   = nn.W*(X-2 )+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        Huprl=[Hu(:),Hpr(:),Hl(:)]';
        Htras=reshape(Huprl(:),nn.hiddensize*3,[]); 
        H=Htras; %Lx3N
    case{'type2sig2'}
        H=TensorReshape(nn,X,nn.shifti,tempH,tempH2,ndata,L,N);    
end

clear tempH;

switch(nn.method{alg_ind})
    case 'ELM'
        [beta] = regressor(H', Y', 0);
    case 'RELM'
        [beta] = regressor(H', Y', nn.C{alg_ind});
    case {'WRELM','TriT2WRELM'}
        [beta] = regressor(H', Y', nn.C{alg_ind});
        e = beta'*H - Y;
        %s = iqr(e)/(2*0.6745);
        %e = sum(abs(e),1);
        s = median(abs(e))/0.6745;
        w = weight_fun(e, nn.wfun{alg_ind}, s,alg_ind);
        [beta] = regressor( repmat(sqrt(w'),1,size(H,1)).*H', repmat(sqrt(w'),1,size(Y,1)).*Y' , nn.C{alg_ind});
    case {'TriT2RELM'} %%%%%%
        [beta] = regressor(H', Y', nn.C{alg_ind});% H':3LxN
        e = beta'*H - Y;% Y:1xN
        s = median(abs(e))/0.6745;
        w = weight_fun(e, nn.wfun{alg_ind}, s,alg_ind);% 3LxN
        [beta] = regressor( repmat(sqrt(w'),1,size(H,1)).*H', repmat(sqrt(w'),1,size(Y,1)).*Y' , nn.C{alg_ind});       
    case 'ORELM'
        [beta] = regressor_alm(H', Y', nn.C{alg_ind}, 20);
    case 'TT2ELM' % tensor type-2 regresss 
      % MP inverse of tensor-2017-05
       % Moore-Penrose Inverse of a tensor I1 x...x In x J1 x...x Jn
        % Sun, L., et al., Moore¨CPenrose inverse of tensors via Einstein product. 
        % Linear and  Multilinear  Algebra, 2016. 64(4): p. 686-698.
        siz1=size(H);
        N=2;M=length(siz1);
        A_tendecomp=MPtensorInvese(N,M,siz1,H);
        beta=tprod(A_tendecomp.MPINV,[1:N,-(1:M-N)],Y'*ones(1,2),[-(1:N),N+1:M]);   
end

nn.time_train(alg_ind) = toc;
if alg_ind<=3
    nn.beta  = beta';
    Y_hat    = nn.beta*H;
else
     nn.beta  = beta;
     Y_hat    = tprod(H,[1:N,-(1:M-N)], beta,[-(1:N),N+1:M]); 
end

if ismember(nn.type,{'c','classification','Classification'})
    [~,label_actual]  = max(Y_hat,[],1);
    [~,label_desired] = max(Y,[],1);
    acc_train = sum(label_actual==label_desired)/ndata;
else
    if alg_ind<=3
        if nn.mapmmPSFlag{nn.DataNameId}==1
            YYhatError=mapminmax('reverse',Y,nn.trainlabelG_PS{nn.DataNameId})...
                -mapminmax('reverse',Y_hat,nn.testlabel_PS{nn.DataNameId});
            normfro = norm(YYhatError,'fro');
        else
            % without mapminmax case
            normfro = norm(Y-Y_hat,'fro');
        end
    else % tensor reg case
        if nn.mapmmPSFlag{nn.DataNameId}==1
            YYhatErrorT=mapminmax('reverse',Y(:),nn.trainlabelG_PS{nn.DataNameId})...
                -mapminmax('reverse',Y_hat(:,1),nn.testlabel_PS{nn.DataNameId});
            normfro = norm(YYhatErrorT,'fro');
        else
            normfro = norm(Y(:)-Y_hat(:,1),'fro');
        end
    end
%     acc_train = sqrt(normfro^2/ndata);
    if alg_ind<=3
        if nn.mapmmPSFlag{nn.DataNameId}==1
            YYhatError=mapminmax('reverse',Y,nn.trainlabelG_PS{nn.DataNameId})...
                -mapminmax('reverse',Y_hat,nn.testlabel_PS{nn.DataNameId});
            acc_train = sqrt(mse(YYhatError));
        else % without mapminmax case
            acc_train = sqrt(mse(Y-Y_hat));
        end
    else % tensor reg case
        if nn.mapmmPSFlag{nn.DataNameId}==1
             YYhatErrorT=mapminmax('reverse',Y(:),nn.trainlabelG_PS{nn.DataNameId})...
                -mapminmax('reverse',Y_hat(:,1),nn.testlabel_PS{nn.DataNameId});
            acc_train = sqrt(mse(YYhatErrorT));
        else
            acc_train = sqrt(mse(Y(:)-Y_hat(:,1)));
        end               
    end
end
nn.trainlabel  = Y_hat;
nn.acc_train   = acc_train;



