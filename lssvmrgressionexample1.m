% 数据XY里随机分解指定个数的训练集和验证集
clc;  
clear
addpath(genpath('./.'));
% X = (-3:0.01:3)';  
% Y = sin(pi.*X+12345*eps)./(pi*X+12345*eps)+0.1.*randn(length(X),1);  

gam = 10;  
sig2 = 0.3;  

type = 'function estimation';  

DName={'bank',...% 1
       'triazines',... % 2
       };
DataNameId      = 2;  
nn.DataNameId   = DataNameId;

% 样本数为偶数
    
% 1='bank', 2='triazines'  
DataName = string(DName(DataNameId))
Iter=1000;
acc_train=zeros(Iter,1);  
acc_test=zeros(Iter,1);  
for It=1:Iter
    switch DataName
         case  'bank' % 3
            bank=load('bank.data');  %8192
            bankIdent1=bank';
            trainNum=4E3;
            bankSize=size(bankIdent1,2);        
            rand_sequence=randperm(bankSize,trainNum);
            rand_seqtest=setdiff(1:bankSize,rand_sequence);
            nn.mapmmPSFlag{DataNameId}=1;
            [traindata,nn.traindata_PS{DataNameId}] = mapminmax(bankIdent1(1:end-1,rand_sequence));
            [trainlabel,nn.trainlabelG_PS{DataNameId}] = mapminmax(bankIdent1(end,rand_sequence));
            [testdata,nn.testdata_PS{DataNameId}] = mapminmax(bankIdent1(1:end-1,rand_seqtest));
            [testlabel,nn.testlabel_PS{DataNameId}] = mapminmax(bankIdent1(end,rand_seqtest));              
        case 'triazines' % 5  186
            trainNum=100;
            nn.mapmmPSFlag{DataNameId}=0;
            triazines=load('triazines.mat','double0');
            rand_sequence=randperm(size(triazines.double0,2),trainNum);
            rand_seqtest=setdiff(1:size(triazines.double0,2),rand_sequence);        
            traindata=triazines.double0(1:end-1,rand_sequence);
            trainlabel=triazines.double0(end,rand_sequence);
            testdata=triazines.double0(1:end-1,rand_seqtest);
            testlabel=triazines.double0(end,rand_seqtest);    
        otherwise
            warning('Wrong datasets, exit.')
            exit
    end

    %[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});  
    %[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','original'});
    trainNum=size(traindata,2); 

    [alpha,b] = trainlssvm({traindata',trainlabel',type,gam,sig2,'RBF_kernel','preprocess'});  

    Yte = simlssvm({traindata',trainlabel',type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},testdata'); 
    Ytr = simlssvm({traindata',trainlabel',type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},traindata');
    if nn.mapmmPSFlag{nn.DataNameId}==1
        YYhatErrorT=mapminmax('reverse',trainlabel,nn.trainlabelG_PS{nn.DataNameId})...
                        -mapminmax('reverse',Ytr(:,1),nn.testlabel_PS{nn.DataNameId});
        acc_train1 = sqrt(mse(YYhatErrorT));
    else
        Yhat1=Ytr(:,1);
        acc_train1 = sqrt(mse(trainlabel(:)-Yhat1));
    end
    if nn.mapmmPSFlag{nn.DataNameId}==1
        YYhatErrorT=mapminmax('reverse',testlabel,nn.testlabel_PS{nn.DataNameId})...
                        -mapminmax('reverse',Yte(:,1),nn.testlabel_PS{nn.DataNameId});
        acc_test1 = sqrt(mse(YYhatErrorT));
    else
        Yhat1=Yte(:,1);
        acc_test1 = sqrt(mse(testlabel(:)-Yhat1));
    end
    acc_train(It,1)=acc_train1;  
    acc_test(It,1)=acc_test1;
end
sprintf('Training Mean  %d  Training Std   %d  Testing Mean  %d Testing  Std %d',...
              mean(acc_train),std(acc_train),mean(acc_test),std(acc_test))
% figure; 
% plotlssvm({X(rand_sequence,:),Y(rand_sequence,:),type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});  
% hold off 

% Xt = (min(X):.1:max(X))';  
% Yt = sin(pi.*Xt+12345*eps)./(pi*Xt+12345*eps)+0.1.*randn(length(Xt),1);  

% hold on;  
% plot(X(rand_seqtest,:),Yt,'r-.');
% legend('Train Function','Train Data')
% hold off  