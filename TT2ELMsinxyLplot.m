% isnoise=0             no noise case
% isnoise=1             {-1,1} case
% isnoise=2             [-1,1] case
% isEXInvalAdd          = 1 add the extended interval
% ExInterval            = 1 extended interval lenght
clc
clear all

addpath(genpath('./.'));

%% data corrupted by noise {-1,1} or [-1,1]

% isnoise   = 0
% isnoise   = 1
% isnoise   = 2
noise_num=[10];% number of noise data

isnoise         = 0
isEXInvalAdd    = 0;
ExInterval      = 0.05;

DataNameId      = 1;  
nn.DataNameId   = DataNameId;

Sincxy1=load('SincxyData.mat','-mat');  
SincxyIdent1=(Sincxy1.Sincxytr)';
trainNum=size(SincxyIdent1,2);
rand_sequence=randperm(trainNum,trainNum);
rand_sequence1=randperm(size(Sincxy1.Sincxyte',2),size(Sincxy1.Sincxyte',2));
SincxyIdent2=(Sincxy1.Sincxyte)';
nn.mapmmPSFlag{DataNameId}=1;
[traindata,nn.traindata_PS{DataNameId}] = mapminmax(SincxyIdent1(1:end-1,rand_sequence));
[trainlabel,nn.trainlabelG_PS{DataNameId}] = mapminmax(SincxyIdent1(end,rand_sequence));
[testdata,nn.testdata_PS{DataNameId}] = mapminmax(SincxyIdent2(1:end-1,rand_sequence1));
[testlabel,nn.testlabel_PS{DataNameId}] = mapminmax(SincxyIdent2(end,rand_sequence1));        

nn.traindata_size                           = size(traindata,2);
nn.testdata_size                            = size(testdata,2);

TrainTesting=[];

method            = {'TT2ELM'};
type              = {'regression','classification'};
DirGen            =  'LatexData';

dir2 = strcat('LatexData\','SincxyLPlot\'); 
if ~(exist(dir2))
    mkdir(dir2)
end     

C=1:100;       
L=25:75;
lenC=length(C);
lenL=length(L);
Cond            = zeros(lenC,lenL);
Acc_test        = Cond;
Acc_train       = Cond;
for ci=1:lenC
    for li=1:lenL
        nn.hiddensize     = L(li);
        nn.type           = type{1};
        nn.inputsize      = size(traindata,1);
        nn.orthogonal     = false;
        nn.nTrainingData  = size(traindata,2);
        nn                   = elm_initialization(nn);
        nn.method{6}         = method{1};

        nn.wfun{6}           = '1';
        nn.scale_method{6}   = 1;
        alg_ind              = 6;
        nn.activefunction{6} = 'type2sig2';
        nn                   = elm_initialization(nn);
        nn.C{6}              = C(ci);
        nn.shifti            = [0.1,0.2]; % pianyi
        if isEXInvalAdd ==1
            [nn, acc_train{6}]   = elm_train(traindata, trainlabel, nn,alg_ind);
            [nn6, acc_test{6}]   = elm_test(testdataEx, testlabelEx, nn, alg_ind);    
        else
            [nn, acc_train{6}]   = elm_train(traindata, trainlabel, nn,alg_ind);
            [nn6, acc_test{6}]   = elm_test(testdata, testlabel, nn, alg_ind); 
        end
        %%%%%%%%%%%%%%%%%%%%%%
        Acc_train(ci,li)     = double(acc_train{6});%training
        Acc_test(ci,li)      = double(acc_test{6});%testing  
    end
end
%% plot the figure
figure(1)
% for ii=1:lenC
%    plot(L',Acc_train(ii,:)') 
%    hold on
% end
plot(L',mean(Acc_train(:,:),1)')
xlabel('L')
ylabel('Error')
%legend({'$2^{-5}$', '$2^{-10}$','$2^{-15}$', '$2^{-20}$','$2^{-25}$', '$2^{-30}$'}, 'Interpreter','latex' );
epsname = strcat('SinCxyCLplotTr', '.eps' );
saveas(gcf,epsname,'epsc2')
figure(2)
% for ii=1:lenC
%    plot(L',Acc_test(ii,:)') 
%    hold on
% end
plot(L',mean(Acc_test(:,:),1)') 
xlabel('L')
ylabel('Error')
%legend({'$2^{-5}$', '$2^{-10}$','$2^{-15}$', '$2^{-20}$','$2^{-25}$', '$2^{-30}$'}, 'Interpreter','latex' );
epsname = strcat('SinCxyCLplotTe', '.eps' );
saveas(gcf,epsname,'epsc2')
