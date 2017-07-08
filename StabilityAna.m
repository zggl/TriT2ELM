
% V2.0  use the cell type to code four alg.
%       add the noise to code four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
% V3.0
%       add the TriT2ELM
% isnoise=0             no noise case
% isEXInvalAdd          = 1 add the extended interval
clc
clear all
addpath(genpath('./.'));

%% data corrupted by noise {-1,1} or [-1,1]

% isnoise   = 0
noise_num=[10];% number of noise data

isnoise         = 0

DName={'sinc',...           % 1
       'AutoMPG',...        % 2
       'bank',...           % 3
       'diabetes',...       % 4
       'triazines',...      % 5
       'NonLinSysIdentify',...% 6
       'Sincxy'              % 7
       };
   
DataNameId      = 5;  
nn.DataNameId   = DataNameId;
DataName = string(DName(DataNameId))
% 样本数为偶数
    
% 1= 'sinc', 2='AutoMPG', 3='bank', 4= 'diabetes',5='triazines' 
% 6='NonLinearSysIdentify', 7= 'Sincxy' 

%% data selection fun

Iter =1E2:100:1E3
nn.mapmmPSFlag{DataNameId}=0;
[traindata,trainlabel,testdata,testlabel,C,L,nn]=StabillityDataFun(DataName,DataNameId,nn);

nn.traindata_size = size(traindata,2);
nn.testdata_size  = size(testdata,2);
method          = {'ELM','RELM','WRELM','TT2ELM'};
type            = {'regression','classification'};
DirGen          = 'StabilityAnas';
Cidata            = zeros(length(Iter),4);
Acc_test        = [];
TrainTimeRec    = [];
TestTimeRec     = [];
Acc_train       = [];
TotalIter       = length(L)*length(L)*Iter;
isEXInvalAdd    = 0;
if DataNameId      == 5
    nn.shifti = [0.2,0.4]; % pianyi
else
    nn.shifti       = [0.01,0.05]; % pianyi
end

Test_result=zeros(length(method),length(Iter));
for Itelen=1:length(Iter)
    for id=1:length(noise_num) % for length(noise_num) different Num noise data
        for li=1:length(L)
            for ci=1:length(C)
                    for ii=1:Iter(Itelen) 
                        nn.hiddensize     = L(li);
                        nn.type           = type{1};
                        nn.inputsize      = size(traindata,1);
                        nn.orthogonal     = false;
                        nn.nTrainingData  = size(traindata,2);
                     %% call main --Five Algs

                        %disp(sprintf('Tatal Iter: %d, Curent Iter:  %d',TotalIter,ii))
                        [nn, acc_train, acc_test,TimeRec1,TimeRec2]=...
                            ELMTypeFun(method,nn,C,ci,traindata,trainlabel,testdata,testlabel,isEXInvalAdd);

                        Acc_train(ii,:,ci,li,id)     = cell2mat(acc_train)';%training
                        Acc_test(ii,:,ci,li,id)      = cell2mat(acc_test)';%testing
                        TrainTimeRec(ii,:,ci,li,id)  = TimeRec1;
                        TestTimeRec(ii,:,ci,li,id)   = TimeRec2;
                    end
                    % end of ii 
            end
            % end of ci
        end
        % end of li
    end
    Cidata(Itelen,:)=squeeze(min(Acc_test(:,:,:,li,id),[],1));% algs x lenC
end
CidataName=strcat('Cidata',DName{DataNameId});
save(strcat(CidataName,'.mat'),'Cidata')
%%
CidataName=strcat('Cidata',DName{DataNameId});
Cidata1=Cidata;
plot(Iter',Cidata1(:,1),'k-.',Iter',Cidata1(:,2),'m-*',Iter',Cidata1(:,3),'r--+',Iter',Cidata1(:,4),'b--o')
xlabel('Iter')
%ylabel('$\frac{\sin(x)}{x}$', 'Interpreter','latex' )
ylabel('Testing error');
legend('ELM','RELM','WRELM','TT2-ELM' );
epsname = strcat(CidataName, '.eps' );
saveas(gcf,epsname,'epsc2')
%MagInset(h1, -1, [0.04 0.045 1 4], [0.05 0.09 5.5 9], {'NW','NW';'SE','SE'});
%% write to tex table. 



