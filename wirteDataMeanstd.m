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