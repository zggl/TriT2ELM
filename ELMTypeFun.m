function [nn, acc_train, acc_test,TimeRec1,TimeRec2]=ELMTypeFun(method,nn,C,ci,traindata,...
                                    trainlabel,testdata,testlabel,isEXInvalAdd)
% V2.0  use the cell type to code four alg.
%       add the noise to code four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
%       testdata's xaxis can be extended via XaxisEx=1 for the four alg.
% V3.0
%       add the TriT2ELM
fprintf('       method    |  Training Acc.  |    Testing Acc.   |   Training Time   |    Condition number  \n\n');
%%%------------------------------------------------------------------------
%%%   ELM / Original ELM
%%%-----------------------------------------------------------------------
nn.method{1}         = method{1};
nn                   = elm_initialization(nn);
alg_ind              = 1;
nn.activefunction{1} = 'sig';
if isEXInvalAdd ==1
    [nn, acc_train{1}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn1, acc_test{1}]   = elm_test(testdataEx, testlabelEx, nn, alg_ind);     
else
    [nn, acc_train{1}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn1, acc_test{1}]   = elm_test(testdata, testlabel, nn, alg_ind);    
end

fprintf('      %5s      |      %.4d      |      %.4d      |      %.4d      |      %.4d      \n',...
    nn.method{1},acc_train{1},acc_test{1},nn.time_train(1),cond(nn1.H,2));

%%%------------------------------------------------------------------------
%%%   RELM / Regularized ELM
%%%------------------------------------------------------------------------

nn.method{2}         = method{2};
nn                   = elm_initialization(nn);
nn.C{2}              = C(ci);
alg_ind              = 2;
nn.activefunction{2} = 'sig';
if isEXInvalAdd ==1
    [nn, acc_train{2}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn2, acc_test{2}]   = elm_test(testdataEx, testlabelEx, nn, alg_ind);    
else
    [nn, acc_train{2}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn2, acc_test{2}]   = elm_test(testdata, testlabel, nn, alg_ind); 
end
fprintf('      %5s      |      %.4d      |      %.4d      |      %.4d      |      %.4d      \n',...
    nn.method{2},acc_train{2},acc_test{2},nn.time_train(2),cond(nn2.H,2));

%%%------------------------------------------------------------------------
%%%   WRELM / Weighted Regularized ELM
%%%------------------------------------------------------------------------

nn.method{3}         = method{3};
nn.wfun {3}          = '1';
nn.scale_method{3}   = 1;
alg_ind              = 3;
nn.activefunction{3} = 'sig';
nn                   = elm_initialization(nn);
nn.C{3}              = C(ci);
if isEXInvalAdd ==1
    [nn, acc_train{3}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn3, acc_test{3}]   = elm_test(testdataEx, testlabelEx, nn, alg_ind);    
else
    [nn, acc_train{3}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn3, acc_test{3}]   = elm_test(testdata, testlabel, nn, alg_ind); 
end
fprintf('      %5s      |      %.4d      |      %.4d      |      %.4d      |      %.4d      \n',...
    nn.method{3},acc_train{3},acc_test{3},nn.time_train(3),cond(nn3.H,2));
% 
%%%------------------------------------------------------------------------
%%%   TensorT2ELM / triangular type-2 Regularized ELM 20170525
% %%%------------------------------------------------------------------------
%nn.DataNameId        = DataName;
nn.method{4}         = method{4};
nn.wfun{4}           = '1';
nn.scale_method{4}   = 1;
alg_ind              = 4;
nn.activefunction{4} = 'type2sig2';
nn                   = elm_initialization(nn);
nn.C{4}              = C(ci);
if isEXInvalAdd ==1
    [nn, acc_train{4}]   = elm_train(traindata, trainlabel, nn,alg_ind);
    [nn4, acc_test{4}]   = elm_test(testdataEx, testlabelEx, nn, alg_ind);    
else
    [nn, acc_train{4}]   = elm_trainforstability(traindata, trainlabel, nn,alg_ind);
    [nn4, acc_test{4}]   = elm_testforstability(testdata, testlabel, nn, alg_ind); 
end
%%%%%%%%%%%%%%%%%%%%%%
fprintf('      %5s     |      %.4d      |      %.4d      |      %.4d      |  \n\n',...
    nn.method{4},acc_train{4},acc_test{4},nn.time_train(4) );
TimeRec1=[nn.time_train(1),nn.time_train(2),nn.time_train(3),nn.time_train(4)];
TimeRec2=[nn1.time_test(1),nn2.time_test(2),nn3.time_test(3),nn4.time_test(4)];
end
