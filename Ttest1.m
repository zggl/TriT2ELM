%% t-test 1
N_1=30;N_2=30;nv1=N_1-1;nv2=N_2-1;
%%% t-test
clear T
for ii=1:size(A,1)% for 8 test benchmark functions
    %%% BFA vs. other 5 alg.
    for jj=1:6 % six alg.
        for kk=1:3 %BFA IBFA FIBFA
            for dim=1:3
                 if kk==jj
                   T(ii,kk,jj,dim)=nan;%'NA'
                else    
                    nv0=(A(kk,2,1,dim)^2/N_1+A(jj,2,1,dim)^2/N_2)^2/(A(kk,2,1,dim)^4/(N_1^2*nv1)+A(jj,2,1,dim)^4/(N_2^2*nv2));
                    T(ii,kk,jj,dim)=(A(jj,1,1,dim)-A(kk,1,1,dim))/sqrt(A(kk,2,1,dim)^2/nv0+A(jj,2,1,dim)^2/nv0);%  8x3x6x3                 
                end               
            end
        end    
    end  
end
size(T)
%% format to 
for ii=1:size(T,1)
    for dim=1:3
        Ttemp1=shiftdim(T(1,1,:,:)); 
        Tt(:,ii)=Ttemp1(:);
    end
end
%% T:8x3x6x3  
%% –¥»Îlatax±Ì∏Ò 7-9=t-test results with respect to BFA, IBFA and FIBFA
laxtabname='ltxtab.tex';
if exist(laxtabname,'file')
    delete(laxtabname);
end
DimMat1=[30:15:60];
DirSixName={'BFA','IBFA','FIBFA','IQBFA','FIQBFA','AQBFA'};
idNumList=[1,3,5,7,9,10,11,12];
f_flag=1;dim_flag=1;
for ii=1:size(A,1)% for 8 test benchmark functions
    if f_flag==1
        strii=strcat('$f_{',num2str(idNumList(ii)),'}$');
    end
    for jj=1:6 % six algorithms        
        for dim=1:3 %three dimension
            if dim==1 % first line
                str2=strcat(strii,' &',DirSixName{jj},' &',num2str(DimMat1(dim),'%d'),' &',num2str(A(ii,1,jj,dim),'%6.4e'),' &',num2str(A(ii,2,jj,dim),'%6.4e'),'&',num2str(AAAaind(jj,dim),'%d'));
                %%% T(ii,kk,jj,dim) ii=8 test benchmark functions, kk=%BFA
                %%% IBFA FIBFA, jj=six alg., dim=[30,45,60]
                str_ttest1=strcat(' &',num2str(T(ii,1,jj,dim),'%.4f'),' &',num2str(T(ii,2,jj,dim),'%.4f'),' &',num2str(T(ii,3,jj,dim),'%.4f'),'\\');
                dlmwrite(laxtabname,strcat(str2,str_ttest1),'delimiter', '','-append');
                dlmwrite(laxtabname, [],'newline','pc','-append');
                f_flag=0;strii='';
            else
                str2=strcat('             &',' &',num2str(DimMat1(dim),'%d'),' &',num2str(A(ii,1,jj,dim),'%6.4e'),' &',num2str(A(ii,2,jj,dim),'%6.4e'),' &',num2str(AAAaind(jj,dim),'%d'));
                %%% T(ii,kk,jj,dim) ii=8 test benchmark functions, kk=%BFA
                %%% IBFA FIBFA, jj=six alg., dim=[30,45,60]
                str_ttest1=strcat(' &',num2str(T(ii,1,jj,dim),'%.4f'),' &',num2str(T(ii,2,jj,dim),'%.4f'),' &',num2str(T(ii,3,jj,dim),'%.4f'),'\\');
                dlmwrite(laxtabname,strcat(str2,str_ttest1),'delimiter', '','-append');
                dlmwrite(laxtabname, [],'newline','pc','-append');
            end        
        end 
    end
    f_flag=1;
    dlmwrite(laxtabname, [],'newline','pc','-append');
    dlmwrite(laxtabname, '\midrule','delimiter', '','newline','pc','-append');
    
end
type(laxtabname)