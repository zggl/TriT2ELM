
% V2.0  use the cell type to code four alg.
%       add the noise to code four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
% V3.0
%       add the TriT2ELM
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

DName={'sinc',...   % 1
       'AutoMPG',... % 2
       'bank',...% 3
       'diabetes',... % 4
       'triazines',... % 5
       'NonLinSysIdentify',... % 6
       'Sincxy' % 7
       };
DataNameId      = 5;  
nn.DataNameId   = DataNameId;

% 样本数为偶数
    
% 1= 'sinc', 2='AutoMPG', 3='bank', 4= 'diabetes',5='triazines' 
% 6='NonLinearSysIdentify', 7= 'Sincxy'   
DataName = string(DName(DataNameId))
Iter=1000;
nn.shifti = [0.01,0.05]; % pianyi
switch DataName
    case 'sinc'  % 1
        [traindata,trainlabel,testdata,testlabel]   = sinc_K(0.4,isEXInvalAdd,ExInterval);
        nn.mapmmPSFlag{DataNameId}=0; % without mapminmax
        C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
        L=[25,30,35];
        style=1; % mean and stdard error
    case  'AutoMPG'  % 2  392x8
        nn.mapmmPSFlag{DataNameId}=1;
        trainNum=300; 
        AutoMPG=load('auto-mpg.data')';
        rand_sequence=randperm(size(AutoMPG,2),trainNum);
        rand_seqtest=setdiff(1:size(AutoMPG,2),rand_sequence);
        [traindata,nn.traindata_PS{DataNameId}] = mapminmax(AutoMPG(1:end-1,rand_sequence),0,0.01);
        [trainlabel,nn.trainlabelG_PS{DataNameId}] = mapminmax(AutoMPG(end,rand_sequence),0,0.01);
        [testdata,nn.testdata_PS{DataNameId}] = mapminmax(AutoMPG(1:end-1,rand_seqtest),0,0.01);
        [testlabel,nn.testlabel_PS{DataNameId}] = mapminmax(AutoMPG(end,rand_seqtest),0,0.01);        
        C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
        L=[40,44,51];
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
        C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
        L=[25,50];       
    case 'diabetes' % 4 
        % diabetes
        nn.mapmmPSFlag{DataNameId}=0;
        diabetes2_data;     %   randomly generate new training and testing data for every trial of simulation
        traindata1=load('diabetes_train')'; %576
        testdata1=load('diabetes_test')';   %192
        traindata=traindata1(1:8,:);
        trainlabel=traindata1(end,:);
        testdata=testdata1(1:8,:);
        testlabel=testdata1(end,:);
        C=[2^(-5),2^(-5),2^(-5),2^(-20),2^(-25),2^(-30)];
        L=[29,29,29];
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
        C=[2^(-5),2^(-5),2^(-5),2^(-20),2^(-25),2^(-30)];
        L=[10,25,35];%-60
    case 'NonLinSysIdentify' % 6
        trainNum=600;
        NonLinSysIdent1=load('NonlinearDataIdentify2.mat','-mat', 'dataNPlant');  
        NonLinSysIdent=(NonLinSysIdent1.dataNPlant)';
        rand_sequence=randperm(size(NonLinSysIdent,2),trainNum);
        rand_seqtest=setdiff(1:size(NonLinSysIdent,2),rand_sequence);
        nn.mapmmPSFlag{DataNameId}=1;
        [traindata,nn.traindata_PS{DataNameId}] = mapminmax(NonLinSysIdent(1:end-1,rand_sequence));
        [trainlabel,nn.trainlabelG_PS{DataNameId}] = mapminmax(NonLinSysIdent(end,rand_sequence));
        [testdata,nn.testdata_PS{DataNameId}] = mapminmax(NonLinSysIdent(1:end-1,rand_seqtest));
        [testlabel,nn.testlabel_PS{DataNameId}] = mapminmax(NonLinSysIdent(end,rand_seqtest));        
        C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
        L=[25,30,35];
    case 'Sincxy' % 7
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
        C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
        L=[25,60,75];       
    otherwise
        warning('Wrong datasets, exit.')
        exit
end
nn.traindata_size                           = size(traindata,2);
nn.testdata_size                            = size(testdata,2);

method            = {'ELM','RELM','WRELM','TriT2WRELM','TriT2RELM','TT2ELM'};
type              = {'regression','classification'};
DirGen            =  'LatexData';
switch DataName
    case 'sinc'
        dir2 = strcat('LatexData\','SinC2\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end
    case 'diabetes'
    % diabetes
        dir2 = strcat('LatexData\','diabetes\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end
    case 'abalone'
        dir2 = strcat('LatexData\','abalone\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end   
    case 'triazines'
        dir2 = strcat('LatexData\','triazines\');  
        if ~(exist(dir2))
            mkdir(dir2)
        end
    case 'CaliforniaHousing'
         dir2 = strcat('LatexData\','CaliforniaHousings\');  
        if ~(exist(dir2))
            mkdir(dir2)
        end       
    case 'servo'
        dir2 = strcat('LatexData\','servos\');  
        if ~(exist(dir2))
            mkdir(dir2)
        end      
    case 'Isatastock'
        dir2 = strcat('LatexData\','Isatastock\');  
        if ~(exist(dir2))
            mkdir(dir2)
        end       
    case 'Yacht'
         dir2 = strcat('LatexData\','Yacht\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end    
    case  'AutoMPG'
        dir2 = strcat('LatexData\','AutoMPG\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end      
    case 'Airfoil'
        dir2 = strcat('LatexData\','Airfoil\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end      
    case 'Kinematic'
        dir2 = strcat('LatexData\','Kinematic\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end
    case 'NonLinSysIdentify'
        dir2 = strcat('LatexData\','NonLinearSysIdentify\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end                
     case 'Sincxy' % 13
        dir2 = strcat('LatexData\','Sincxy\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end
     case  'bank' % 14
        dir2 = strcat('LatexData\','Bank\'); 
        if ~(exist(dir2))
            mkdir(dir2)
        end             
     otherwise
        warning('Wrong datasets. exit.')
        exit
end

Cond            = zeros(Iter,6);

Acc_test        = Cond;
TrainTimeRec    = Cond;
TestTimeRec     = Cond;
Acc_train       = Cond;
TotalIter       = length(L)*length(L)*Iter;

for id=1:length(noise_num) % for length(noise_num) different Num noise data
    corrupt_indy=[];
    randinterv=[];
    randbinary=[];
    ind1=[];
    ind2=[];
    trainlabel_ori = trainlabel;
    for li=1:length(L)
        for ci=1:length(C)
                for ii=1:Iter  
                    if isnoise==1
                        % {-1,1} case
                        corrupt_indy                = sort(randi([1,nn.traindata_size],noise_num(id),1));
                        randbinary                  = randi([-1,1],noise_num(id),1)';
                        ind1                        = setdiff(1:nn.traindata_size,corrupt_indy);
                        corrupt_bivalue             = sign(randbinary+eps);
                        trainlabel(corrupt_indy)    = trainlabel_ori(corrupt_indy)+ corrupt_bivalue;
                        trainlabel(ind1)            = trainlabel_ori(ind1);  
                    elseif isnoise==2
                        %  [-1,1] case
                        corrupt_indy                = sort(randi([1,nn.traindata_size],noise_num(id),1));
                        randinterv                  = (2*rand(noise_num(id),1)-1)';
                        ind2                        = setdiff(1:nn.traindata_size,corrupt_indy);
                        trainlabel(corrupt_indy)    = trainlabel_ori(corrupt_indy)+ randinterv;
                        trainlabel(ind2)            = trainlabel_ori(ind2);
                    end
                    if isEXInvalAdd == 1
                        minmaxEx=minmax(testdata)+[-ExInterval,ExInterval];
                        testdataEx=linspace(minmaxEx(1),minmaxEx(2),nn.testdata_size+200);
                        testlabelEx=sin(testdataEx/pi);
                    end
                    nn.hiddensize     = L(li);
                    nn.type           = type{1};
                    nn.inputsize      = size(traindata,1);
                    nn.orthogonal     = false;
                    nn.nTrainingData  = size(traindata,2);
                 %% call main --Five Algs

                    disp(sprintf('Tatal Iter: %d, Curent Iter:  %d',TotalIter,ii))
                    CL_SincTriWRELMRanT2V3 %% 
                    
                    Cond(ii,:,ci,li,id)          = cond_numer;
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
%% write to tex table. 
% RMSE with respect to ELM RELM　WRELM　TriT2WRELM
% preparing data for tex write
% style =0; % mean type
% style=1;  % mean and std type
laxtabname2=strcat(dir2,'ltxtab2.tex');
if exist(laxtabname2,'file')
    delete(laxtabname2);
end
switch style
    case 0
        Tabletil2=strcat('& ',repmat('  Training   &   Testing   &',1,...
            length(noise_num)),'    Training   &   Testing \\');
        dlmwrite(laxtabname2,Tabletil2,'delimiter', '','-append');
        dlmwrite(laxtabname2, [],'newline','pc','-append');
        dlmwrite(laxtabname2,'\hline','delimiter', '','-append');
        % Table structure:  L x noise_num
        for id=1:length(noise_num) % for length(noise_num) different Num noise data
            for li=1:length(L) % % L x noise_num: for different five alg. x 6 ci=6x6
                Cidata=squeeze(mean(Acc_test(:,:,:,li,id),[],1)); 
                Cidata1=squeeze(mean(Acc_train(:,:,:,li,id),[],1));
                % under different Ci=6
                     for ii=1:length(method) % for six alg.
                        strii = method{ii};
                        str2  = strcat(strii,'   &', num2str(Cidata1(ii,1),'%.2d'),...
                                          '   &', num2str(Cidata(ii,1),'%.2d'),...
                                          '   &', num2str(Cidata1(ii,2),'%.2d'),...
                                          '   &', num2str(Cidata(ii,2),'%.2d'),...
                                          '   &', num2str(Cidata1(ii,3),'%.2d'),...
                                          '   &', num2str(Cidata(ii,3),'%.2d'),...
                                           '   &', num2str(Cidata1(ii,4),'%.2d'),...
                                          '   &', num2str(Cidata(ii,4),'%.2d'),'\\');
                        dlmwrite(laxtabname2,str2,'delimiter', '','-append');
                        dlmwrite(laxtabname2, [],'newline','pc','-append');
                    end
            end
        end
        dlmwrite(laxtabname2,'\bottomrule','delimiter', '','-append');       
    case 1
        Tabletil2=strcat('& ','  Mean   &   Std   &','    Mean   &   Std \\');
        dlmwrite(laxtabname2,Tabletil2,'delimiter', '','-append');
        dlmwrite(laxtabname2, [],'newline','pc','-append');
        dlmwrite(laxtabname2,'\hline','delimiter', '','-append');
        % Table structure:  L x noise_num
        for id=1:length(noise_num) % for length(noise_num) different Num noise data
            for li=1:length(L) % % L x noise_num: for different five alg. x 6 ci=6x6
                Cidata=squeeze(min(Acc_test(:,:,:,li,id),[],1)); 
                Cidata1=squeeze(min(Acc_train(:,:,:,li,id),[],1));
                % under different Ci=6
                     for ii=1:length(method) % for six alg.
                        strii = method{ii};
                        str2  = strcat(strii,'   &', num2str(Cidata1(ii,1),'%.2d'),...
                                          '   &', num2str(Cidata(ii,1),'%.2d'),...
                                          '   &', num2str(Cidata1(ii,2),'%.2d'),...
                                          '   &', num2str(Cidata(ii,2),'%.2d'),...
                                          '   &', num2str(Cidata1(ii,3),'%.2d'),...
                                          '   &', num2str(Cidata(ii,3),'%.2d'),...
                                           '   &', num2str(Cidata1(ii,4),'%.2d'),...
                                          '   &', num2str(Cidata(ii,4),'%.2d'),'\\');
                        dlmwrite(laxtabname2,str2,'delimiter', '','-append');
                        dlmwrite(laxtabname2, [],'newline','pc','-append');
                    end
            end
        end
        dlmwrite(laxtabname2,'\bottomrule','delimiter', '','-append');          
end


switch DataName
    case 'NonLinSysIdentify' % 12-save and plot the last results
        plot(NonLinSysIdent(end,rand_seqtest),'--');
        hold on 
        plot(nn6.testlabel);
        xlabel('$k$', 'Interpreter','latex' )
        %ylabel('$\frac{\sin(x)}{x}$', 'Interpreter','latex' )
        ylabel('$\hat y_p(k),y_p(k)$', 'Interpreter','latex' );

        legend({'$y_p(k)$', '$\hat y_p(k)$'}, 'Interpreter','latex' );
        epsname = strcat('nonlinearfunTstRsluts', '.eps' );
        saveas(gcf,epsname,'epsc2')
end



