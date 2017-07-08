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

DataNameId      = 3; 

nn.DataNameId   = DataNameId;

nn.mapmmPSFlag{DataNameId}=1;
trainNum=320; 
AutoMPG=load('auto-mpg.data')';
rand_sequence=randperm(size(AutoMPG,2),trainNum);
rand_seqtest=setdiff(1:size(AutoMPG,2),rand_sequence);
[traindata,nn.traindata_PS{DataNameId}] = mapminmax(AutoMPG(1:end-1,rand_sequence));
[trainlabel,nn.trainlabelG_PS{DataNameId}] = mapminmax(AutoMPG(end,rand_sequence));
[testdata,nn.testdata_PS{DataNameId}] = mapminmax(AutoMPG(1:end-1,rand_seqtest));
[testlabel,nn.testlabel_PS{DataNameId}] = mapminmax(AutoMPG(end,rand_seqtest));
      
nn.traindata_size                           = size(traindata,2);
nn.testdata_size                            = size(testdata,2);

TrainTesting=[];

method            = {'TT2ELM'};
type              = {'regression','classification'};
DirGen            =  'LatexData';

dir2 = strcat('LatexData\','AutoMPGPlot\'); 
if ~(exist(dir2))
    mkdir(dir2)
end     

C=1:100;       
L=5:55;
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
epsname = strcat('AutoMPGplotTr', '.eps' );
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
epsname = strcat('AutoMPGplotTe', '.eps' );
saveas(gcf,epsname,'epsc2')
