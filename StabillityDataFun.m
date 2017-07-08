function [traindata,trainlabel,testdata,testlabel,C,L,nn]=StabillityDataFun(DataName,DataNameId,nn)
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
            %C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
            C=2^(-5)
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
            %C=[2^(-5),2^(-10),2^(-15),2^(-20),2^(-25),2^(-30)];
            C=2^(-5)
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
            C=[2^(-5)];
            L=[10];%-60
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
end