function H=TensorReshape(nn,X,shifti,tempH,tempH2,ndata,L,N)     
    tempHpr = nn.W*(X-shifti(1) )+ repmat(nn.b,1,ndata);
    tempHpr2  = nn.W1*(X-shifti(1) )+ repmat(nn.b,1,ndata);
    tempHl  = nn.W1*(X-shifti(2) )+ repmat(nn.b,1,ndata);
    tempHl2  = nn.W*(X-shifti(2) )+ repmat(nn.b,1,ndata);
    
    H(:,:,:,1) = reshape(exp(-([tempH,tempH2]').^2)/2.5,L,2,N);
    H(:,:,:,2) = reshape(exp(-([tempHpr,tempHpr2]').^2)/2.5,L,2,N);
    H(:,:,:,3) = reshape(exp(-([tempHl,tempHl2]').^2)/2.5,L,2,N); 
end