function [nn, acc_test] = elm_test(X,Y, nn, alg_ind)
tic
ndata        = size(X, 2);
tempH        = nn.W*X + repmat(nn.b,1,ndata);
tempH2        = nn.W1*X + repmat(nn.b,1,ndata);
[N,L]=size(tempH);
switch lower(nn.activefunction{alg_ind})
    case{'s','sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
%         cond(H,2)
    case{'t','tanh'}
        H = tanh(tempH);
        cond(H,2)
    case{'r','rbf'} 
        tempH=-((bsxfun(@minus,X, nn.W)).^2);   
        H=exp(tempH./15);
%         cond(H,2)
    case{'t2s','t2sig','t2sigmoid'}
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr = nn.W*(X-1 )+ repmat(nn.b,1,ndata);
        tempHl = nn.W*(X-2 )+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        H=(Hu+Hpr+Hl)/3;
     case{'t2srnd','t2sigrnd','t2sigmoidrnd'}
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr = nn.W*(X-nn.sigran_mean(1) )+ repmat(nn.b,1,ndata);
        tempHl = nn.W*(X-nn.sigran_mean(2) )+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        H=(Hu+Hpr+Hl)/3;       
%         cond(H,2)
    case{'t2s2','t2sig2','t2sigmoid2'}
        Hu = 1 ./ (1 + exp(-tempH));
        tempHpr        = nn.W*(X-1 )+ repmat(nn.b,1,ndata);
        tempHl        = nn.W*(X-2 )+ repmat(nn.b,1,ndata);
        Hpr = 1 ./ (1 + exp(-tempHpr));
        Hl = 1 ./ (1 + exp(-tempHl));
        Huprl=[Hu(:),Hpr(:),Hl(:)]';
        Htras=reshape(Huprl(:),nn.hiddensize*3,[]); 
        H=Htras; %Lx3N
    case{'type2sig2'}
        H=TensorReshape(nn,X,nn.shifti ,tempH,tempH2,ndata,L,N);        
end
if alg_ind<=5
    nn.H=H;
    Y_hat    = nn.beta*H;
else
    N=2;M=length(size(H));
    nn.H=H;
    Y_hat    = tprod(H,[1:N,-(1:M-N)],nn.beta,[-(1:N),N+1:M]);    
end
nn.time_test(alg_ind) = toc;

if ismember(nn.type,{'c','classification','Classification'})
    [~,label_actual]  = max(Y_hat,[],1);
    [~,label_desired] = max(Y,[],1);
    acc_test = sum(label_actual==label_desired)/ndata;
else
    if alg_ind<=5
        if nn.mapmmPSFlag{nn.DataNameId}==1
            Y1=mapminmax('reverse',Y(:),nn.trainlabelG_PS{nn.DataNameId});
            Yhat1=mapminmax('reverse',Y_hat(:,1),nn.testlabel_PS{nn.DataNameId});
            normfro = norm(Y1-Yhat1,'fro');
        else
            % without mapminmax case
            Yhat1=Y_hat;
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
    if alg_ind<=5
        if nn.mapmmPSFlag{nn.DataNameId}==1
            Y1=mapminmax('reverse',Y(:),nn.trainlabelG_PS{nn.DataNameId});
            Yhat1=mapminmax('reverse',Y_hat(:,1),nn.testlabel_PS{nn.DataNameId});
            acc_test = sqrt(mse(Y1-Yhat1));
        else % without mapminmax case
            Yhat1=Y_hat(:,1);
            acc_test = sqrt(mse(Y-Y_hat));
        end
    else % tensor reg case        
        if nn.mapmmPSFlag{nn.DataNameId}==1
            Y1=mapminmax('reverse',Y(:),nn.trainlabelG_PS{nn.DataNameId});
            Yhat1=mapminmax('reverse',Y_hat(:,1),nn.testlabel_PS{nn.DataNameId});
            acc_test = sqrt(mse(Y1-Yhat1));
        else
            Yhat1=Y_hat(:,1);
            acc_test = sqrt(mse(Y(:)-Y_hat(:,1)));
        end
    end
end
nn.testlabel  = Yhat1;
nn.acc_test   = acc_test;




