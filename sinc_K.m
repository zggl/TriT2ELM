function [traindata,trainlabel,testdata,testlabel] = sinc_K(percent,isEXInvalAdd,ExInterval)

m          = 600;
X1         = 20*rand(1,m)-10;
X1(X1==0)  = 1e-8;
traindata     = X1;
trainlabel = sin(X1)./(X1);
%trainlabel = trainlabel+0.4*rand(size(trainlabel))-0.2;  %[-0.2,0.2]

% x = zeros(1,m);
% p = randperm(m);
% k = floor(m*percent);
% x(p(1:k/2))   = ones(k/2,1);
% x(p(k/2+1:k)) = ones(k/2,1)-2;
% x(p(1:k))     = 2*rand(k,1)-1;
% 
% trainlabel    = trainlabel+x;
if isEXInvalAdd==1
    X2 =sort([linspace(-10-ExInterval,-10,10),-10+20*rand(1,1E4), linspace(10,10+ExInterval,10)]);
else
    X2 =-10:0.01:10;
end
X2(X2==0) = 1e-8;
if mod(length(X2),2)~=0
    testdata  = X2(1:end-1);
    testlabel = sin(X2(1:end-1))./(X2(1:end-1));
else
    testdata  = X2;
    testlabel = sin(X2)./(X2);
end




%     save('datasets','traindata','trainlabel','testdata','testlabel');











