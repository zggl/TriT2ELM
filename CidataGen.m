for Itelen=2:length(Iter)
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
    Cidata(:,Itelen)=min(squeeze(min(Acc_test(:,:,:,li,id),[],1)),[],2);% algs x lenC
end