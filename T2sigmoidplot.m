x=-10:0.1:10;
sigmoid=@(x)1./(1+exp(-x));
y = sigmoid(x);%u
y1 = sigmoid(x-1);%pr
y2 = sigmoid(x-2);%l
x2=2;
y2u = sigmoid(x2);%u
y2pr = sigmoid(x2-1);%pr
y2l = sigmoid(x2-2);%l
figure(1)
plot(x,y,x,y1,x,y2,x2,y2l,'.',x2,y2pr,'.',x2,y2u,'.')

strl=sprintf('(%d,%0.1e)',x2,y2l)
text(x2,y2l,strcat('\rightarrow',strl))
strpr=sprintf('(%d,%0.1e)',x2,y2pr)
text(x2,y2pr,strcat('\rightarrow',strpr))
stru=sprintf('(%d,%0.1e)',x2,y2u)
text(x2,y2u,strcat('\rightarrow',stru))
epsname=strcat('T2sigmoidmemfun','.eps' ); 
saveas(gcf,epsname,'epsc2')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
plot([y2l;y2pr;y2u],[0;1;0])
line([y2l;y2pr;y2u],[0;1;0])
xlim([0,1])
epsname=strcat('TriangularType1MF','.eps' ); 
saveas(gcf,epsname,'epsc2')


