%{
Copyright (C) 2015 University of Electronic Science and Technology of China

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}
% main 
clear all;clc;
%load your traing data here
load('')
%% Normalize
x1 = train_source_alb2(:,1:end-2)/255;
x2 = train_target_alb2(:,1:end-2)/255;

x1_mean = mean(x1,1);
x1 = (x1 - repmat(x1_mean,size(x1,1),1));

x2_mean = mean(x2,1);
x2 = (x2 - repmat(x2_mean,size(x2,1),1));

%% dimension/neuron numbers settings
n = size(x1,2);
% identity dimension
d = 2200;
% age dimension 
p = 600;
% noise
q = 200;
% bridge dimension
k = 500;

%% Identity 1
nn.BO1=zeros(1,n);
nn.K1=cell(1,2);
nn.K1{1}=0.01*randn(n,d);
nn.K1{2}=0.01*randn(d,n);
nn.BI1=zeros(1,d);
nn.P1=cell(1,2);
nn.P1{1}=0.01*randn(n,p);
nn.P1{2}=0.01*randn(p,n);
nn.BA1=zeros(1,p);
nn.Q1=cell(1,2);
nn.Q1{1}=0.01*randn(n,q);
nn.Q1{2}=0.01*randn(q,n);
nn.BN1=zeros(1,q);

%% Identity 2
nn.BO2=zeros(1,n);
nn.K2=cell(1,2);
nn.K2{1}=nn.K1{1};
nn.K2{2}=nn.K1{2};
nn.BI2=zeros(1,d);
nn.P2=cell(1,2);
nn.P2{1}=nn.P1{1};
nn.P2{2}=nn.P1{2};
nn.BA2=zeros(1,p);
nn.Q2=cell(1,2);
nn.Q2{1}=nn.Q1{1};
nn.Q2{2}=nn.Q1{2};
nn.BN2=zeros(1,q);

%% Bridge
nn.H1=cell(1,2);
nn.H1{1}=0.01*randn(p, k);
nn.H1{2}=0.01*randn(k, p);
nn.H2=cell(1,2);
nn.H2{1}=0.01*randn(p, k);
nn.H2{2}=0.01*randn(k, p);
nn.B1=zeros(1,k);
nn.B2=zeros(1,k);

%% for sparsity
nn.sI1=zeros(1,d);
nn.sA1=zeros(1,p);
nn.sN1=zeros(1,q);
nn.sI2=zeros(1,d);
nn.sA2=zeros(1,p);
nn.sN2=zeros(1,q);

%% momentum
vK1{1}=zeros(size(nn.K1{1}));vK1{2}=zeros(size(nn.K1{2}));
vP1{1}=zeros(size(nn.P1{1}));vP1{2}=zeros(size(nn.P1{2}));
vK2{1}=zeros(size(nn.K2{1}));vK2{2}=zeros(size(nn.K2{2}));
vP2{1}=zeros(size(nn.P2{1}));vP2{2}=zeros(size(nn.P2{2}));
vQ1{1}=zeros(size(nn.Q1{1}));vQ1{2}=zeros(size(nn.Q1{2}));
vQ2{1}=zeros(size(nn.Q2{1}));vQ2{2}=zeros(size(nn.Q2{2}));

vBO1=zeros(size(nn.BO1));vBI1=zeros(size(nn.BI1));vBA1=zeros(size(nn.BA1));
vBO2=zeros(size(nn.BO2));vBI2=zeros(size(nn.BI2));vBA2=zeros(size(nn.BA2));
vBN1=zeros(size(nn.BN1));vBN2=zeros(size(nn.BN2));

vK1_2{1}=zeros(size(nn.K1{1}));vK1_2{2}=zeros(size(nn.K1{2}));
vP1_2{1}=zeros(size(nn.P1{1}));vP1_2{2}=zeros(size(nn.P1{2}));
vK2_2{1}=zeros(size(nn.K2{1}));vK2_2{2}=zeros(size(nn.K2{2}));
vP2_2{1}=zeros(size(nn.P2{1}));vP2_2{2}=zeros(size(nn.P2{2}));

vH1{1}=zeros(size(nn.H1{1}));vH1{2}=zeros(size(nn.H1{2}));
vH2{1}=zeros(size(nn.H2{1}));vH2{2}=zeros(size(nn.H2{2}));

vBO1_2=zeros(size(nn.BO1));vBI1_2=zeros(size(nn.BI1));vBA1_2=zeros(size(nn.BA1));
vBO2_2=zeros(size(nn.BO2));vBI2_2=zeros(size(nn.BI2));vBA2_2=zeros(size(nn.BA2));
vB1=zeros(size(nn.B1));vB2=zeros(size(nn.B2));
%% other parameters
nn.s='linear';
nn.f='sigm';  
%bridge activ func
nn.fb='sigm';
nn.fo='sigm';

nn.alpha_ae=0.0001;
nn.alpha=0.0001;

opts.batchsize=10;
opts.numbatches=size(x1,1)/opts.batchsize;
maxEpoch=300;

%momentum,etc.
initialmomentum=0.9;
finalmomentum=0.9;
momentum2=0.9;
weightPenaltyL1=0;
weightPenaltyL2=0;
wL1=0;
wL2=0;
nn.sparsityPenaltyI=0;
nn.sparsityPenaltyA=0;
nn.sparsityPenaltyN=0;
nn.sparsityTarget=0.05;
nn.noise=0;

L_train=zeros(1,maxEpoch);

L_A=zeros(1,maxEpoch);
L_I=zeros(1,maxEpoch);

fid = fopen('decom_err.txt','a');
%% 3 steps, feedforward, gradient compute and update parameters
for j=1:maxEpoch
    
    kk=randperm(size(x1,1));
    if j<5
        momentum=initialmomentum;
    else
        momentum=finalmomentum;
    end

for l=1:opts.numbatches
    batch_x1 = x1(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
    batch_x2 = x2(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
    
    if(nn.noise ~= 0)            
        batch_corr_x1 = batch_x1.*(rand(size(batch_x1))>nn.noise);
        batch_corr_x2 = batch_x2.*(rand(size(batch_x2))>nn.noise);
    else
        batch_corr_x1 = batch_x1;
        batch_corr_x2 = batch_x2;
    end
    
    nn=nnff(nn,batch_corr_x1,batch_corr_x2,batch_x1,batch_x2);
    m=size(batch_x1,1);

%%%%%%%%%%%%%%%%%%% 1-step is now %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% first age
%K1{2}
if strcmp(nn.s,'sigm')
    deri=nn.batch_x1_p.*(1-nn.batch_x1_p);
elseif strcmp(nn.s,'linear')
    deri=1;
elseif strcmp(nn.s,'tanh_opt')
    deri=1 - nn.batch_x1_p.^2;
end
d1{1}=-(batch_x1-nn.batch_x1_p).*deri;
dK1{2}=nn.I1'*d1{1}/m;

%P1{2}
dP1{2}=nn.A1'*d1{1}/m;

%Q1{2}
dQ1{2}=nn.N1'*d1{1}/m;

%K1{1}
if strcmp(nn.f,'sigm')
    deri=nn.I1.*(1-nn.I1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.I1.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyI>0)
    pi = repmat(nn.sI1, m, 1);
    sparsityError = nn.sparsityPenaltyI * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d1{2}=(d1{1}*nn.K1{2}'+sparsityError).*deri;
dK1{1}=batch_corr_x1'*d1{2}/m;

%P1{1}
%1.
if strcmp(nn.f,'sigm')
    deri=nn.A1.*(1-nn.A1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A1.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyA>0)
    pi = repmat(nn.sA1, m, 1);
    sparsityError = nn.sparsityPenaltyA * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d1{3}=(d1{1}*nn.P1{2}'+sparsityError).*deri;
dP1{1}=batch_corr_x1'*d1{3}/m;

%Q1{1}
if strcmp(nn.f,'sigm')
    deri=nn.N1.*(1-nn.N1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.N1.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyN>0)
    pi = repmat(nn.sN1, m, 1);
    sparsityError = nn.sparsityPenaltyN * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d1{4}=(d1{1}*nn.Q1{2}'+sparsityError).*deri;
dQ1{1}=batch_corr_x1'*d1{4}/m;

%% second age
%K2{2}
if strcmp(nn.s,'sigm')
    deri=nn.batch_x2_p.*(1-nn.batch_x2_p);
elseif strcmp(nn.s,'linear')
    deri=1;
elseif strcmp(nn.s,'tanh_opt')
    deri=1 - nn.batch_x2_p.^2;
end
d2{1}=-(batch_x2-nn.batch_x2_p).*deri;
dK2{2}=nn.I2'*d2{1}/m;

%K2{1}
if strcmp(nn.f,'sigm')
    deri=nn.I2.*(1-nn.I2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.I2.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyI>0)
    pi = repmat(nn.sI2, m, 1);
    sparsityError = nn.sparsityPenaltyI * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d2{2}=(d2{1}*nn.K2{2}'+sparsityError).*deri;
dK2{1}=batch_corr_x2'*d2{2}/m;

%P2{1}
%1.
if strcmp(nn.f,'sigm')
    deri=nn.A2.*(1-nn.A2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A2.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyA>0)
    pi = repmat(nn.sA2, m, 1);
    sparsityError = nn.sparsityPenaltyA * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d2{3}=(d2{1}*nn.P2{2}'+sparsityError).*deri;
dP2{1}=batch_corr_x2'*d2{3}/m;

%P2{2}
dP2{2}=nn.A2'*d2{1}/m;

%Q2{2}
dQ2{2}=nn.N2'*d2{1}/m;

%Q2{1}
if strcmp(nn.f,'sigm')
    deri=nn.N2.*(1-nn.N2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.N2.^2;
end
%if add Sparsity
if(nn.sparsityPenaltyN>0)
    pi = repmat(nn.sN2, m, 1);
    sparsityError = nn.sparsityPenaltyN * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
else
    sparsityError=0;
end
d2{4}=(d2{1}*nn.Q2{2}'+sparsityError).*deri;
dQ2{1}=batch_corr_x2'*d2{4}/m;

%bias gradient compute
%BO1
dBO1=sum(d1{1})/m;
%BI1
dBI1=sum(d1{2})/m;
%BA1
dBA1=sum(d1{3})/m;
%BN1
dBN1=sum(d1{4})/m;
%BO2
dBO2=sum(d2{1})/m;
%BI2
dBI2=sum(d2{2})/m;
%BA2
dBA2=sum(d2{3})/m;
%BN2
dBN2=sum(d2{4})/m;

%1-step update param
for i=1:2
    %L1   
    W = nn.K1{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dK1{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vK1{i} = momentum*vK1{i} + dW;
        dW=vK1{i};
    end
             
    nn.K1{i} = nn.K1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.P1{i};   
    L1=W./sqrt(W.^2+0.000001);        
    
    dW = dP1{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vP1{i} = momentum*vP1{i} + dW;
        dW=vP1{i};
    end
             
    nn.P1{i} = nn.P1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.K2{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dK2{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vK2{i} = momentum*vK2{i} + dW;
        dW=vK2{i};
    end
             
    nn.K2{i} = nn.K2{i} - dW;
end

for i=1:2
    %L1   
    W = nn.P2{i};   
    L1=W./sqrt(W.^2+0.000001);        
    
    dW = dP2{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vP2{i} = momentum*vP2{i} + dW;
        dW=vP2{i};
    end
             
    nn.P2{i} = nn.P2{i} - dW;
end
for i=1:2
    %L1   
    W = nn.Q1{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dQ1{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vQ1{i} = momentum*vQ1{i} + dW;
        dW=vQ1{i};
    end
             
    nn.Q1{i} = nn.Q1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.Q2{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dQ2{i} + weightPenaltyL1 * L1 + weightPenaltyL2 * W;
    dW = nn.alpha_ae * dW;
    
    if(momentum>0)
        vQ2{i} = momentum*vQ2{i} + dW;
        dW=vQ2{i};
    end
             
    nn.Q2{i} = nn.Q2{i} - dW;
end

db = nn.alpha_ae * dBO1;
if(momentum>0)
    vBO1= momentum*vBO1 + db;
    db=vBO1;
end
nn.BO1=nn.BO1-db;

db = nn.alpha_ae * dBO2;
if(momentum>0)
    vBO2= momentum*vBO2 + db;
    db=vBO2;
end
nn.BO2=nn.BO2-db;

db = nn.alpha_ae * dBI1;
if(momentum>0)
    vBI1= momentum*vBI1 + db;
    db=vBI1;
end
nn.BI1=nn.BI1-db;

db = nn.alpha_ae * dBI2;
if(momentum>0)
    vBI2= momentum*vBI2 + db;
    db=vBI2;
end
nn.BI2=nn.BI2-db;

db = nn.alpha_ae * dBA1;
if(momentum>0)
    vBA1= momentum*vBA1 + db;
    db=vBA1;
end
nn.BA1=nn.BA1-db;

db = nn.alpha_ae * dBA2;
if(momentum>0)
    vBA2= momentum*vBA2 + db;
    db=vBA2;
end
nn.BA2=nn.BA2-db;

db = nn.alpha_ae * dBN1;
if(momentum>0)
    vBN1= momentum*vBN1 + db;
    db=vBN1;
end
nn.BN1=nn.BN1-db;

db = nn.alpha_ae * dBN2;
if(momentum>0)
    vBN2= momentum*vBN2 + db;
    db=vBN2;
end
nn.BN2=nn.BN2-db;
%%%%%%%%%%%%%%%%%%% 1-step is over %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear dK1 dK2 dP2 dP1 dBA2 dBI2 dBO2 dBO1 dBI1 dBA1 dBN1 dBN2 dQ1 dQ2
clear db dW W L1

%% 2-step is now
nn=nnff(nn,batch_corr_x1,batch_corr_x2,batch_x1,batch_x2);
%K1{2}
if strcmp(nn.s,'sigm')
    deri=nn.batch_x1_p2.*(1-nn.batch_x1_p2);
elseif strcmp(nn.s,'linear')
    deri=1;
elseif strcmp(nn.s,'tanh_opt')
    deri=1 - nn.batch_x1_p2.^2;
end
d_re1{1}=-(batch_x1-nn.batch_x1_p2).*deri;
dK1{2}=nn.I1'*d_re1{1}/m;

%P1{2}
dP1{2}=nn.A1_p'*d_re1{1}/m;

%K1{1}
%1.
if strcmp(nn.f,'sigm')
    deri=nn.I1.*(1-nn.I1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.I1.^2;
end
d_re1{2}=(d_re1{1}*nn.K1{2}').*deri;

dK1_1{1}=batch_corr_x1'*d_re1{2}/m;
%2.
d_equ1=-(nn.I2-nn.I1).*deri;
dK1_1{2}=batch_corr_x1'*d_equ1/m;

dK1{1}=dK1_1{1}+dK1_1{2};


%K2{1}
%1.
if strcmp(nn.s,'sigm')
    deri=nn.batch_x2_p2.*(1-nn.batch_x2_p2);
elseif strcmp(nn.s,'linear')
    deri=1;
elseif strcmp(nn.s,'tanh_opt')
    deri=1 - nn.batch_x2_p2.^2;
end
d_re2{1}=-(batch_x2-nn.batch_x2_p2).*deri;

if strcmp(nn.f,'sigm')
    deri=nn.I2.*(1-nn.I2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.I2.^2;
end
d_re2{2}=(d_re2{1}*nn.K2{2}').*deri;
dK2_1{1}=batch_corr_x2'*d_re2{2}/m;
%2.
d_equ2=(nn.I2-nn.I1).*deri;
dK2_1{2}=batch_corr_x2'*d_equ2/m;

dK2{1}=dK2_1{1}+dK2_1{2};

%P1{1}
%1.
if strcmp(nn.fo,'sigm')
    deri=nn.A2_p.*(1-nn.A2_p);
elseif strcmp(nn.fo,'linear')
    deri=1;
elseif strcmp(nn.fo,'tanh_opt')
    deri=1 - nn.A2_p.^2;
end
d_bri1{1}=-(nn.A2-nn.A2_p).*deri;

if strcmp(nn.fb,'sigm')
    deri=nn.Y1.*(1-nn.Y1);
elseif strcmp(nn.fb,'linear')
    deri=1;
elseif strcmp(nn.fb,'tanh_opt')
    deri=1 - nn.Y1.^2;
end
d_bri1{2}=(d_bri1{1}*nn.H1{2}').*deri;

if strcmp(nn.f,'sigm')
    deri=nn.A1.*(1-nn.A1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A1.^2;
end
d_bri1{3}=(d_bri1{2}*nn.H1{1}').*deri;

dP1_1{1}=batch_corr_x1'*d_bri1{3}/m;
%2.
d_bri2_ano=(nn.A1-nn.A1_p).*deri;
dP1_1{2}=batch_corr_x1'*d_bri2_ano/m;
%3.
if strcmp(nn.fo,'sigm')
    deri=nn.A2_p.*(1-nn.A2_p);
elseif strcmp(nn.fo,'linear')
    deri=1;
elseif strcmp(nn.fo,'tanh_opt')
    deri=1 - nn.A2_p.^2;
end
d_re2{3}=(d_re2{1}*nn.P2{2}').*deri;

if strcmp(nn.fb,'sigm')
    deri=nn.Y1.*(1-nn.Y1);
elseif strcmp(nn.fb,'linear')
    deri=1;
elseif strcmp(nn.fb,'tanh_opt')
    deri=1 - nn.Y1.^2;
end
d_re2{4}=(d_re2{3}*nn.H1{2}').*deri;

if strcmp(nn.f,'sigm')
    deri=nn.A1.*(1-nn.A1);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A1.^2;
end
d_re2{5}=(d_re2{4}*nn.H1{1}').*deri;
dP1_1{3}=batch_corr_x1'*d_re2{5}/m;

dP1{1}=dP1_1{1}+dP1_1{2}+dP1_1{3};

%K2{2}
dK2{2}=nn.I2'*d_re2{1}/m;

%P2{2}
dP2{2}=nn.A2_p'*d_re2{1}/m;

%P2{1}
%1.
if strcmp(nn.fo,'sigm')
    deri=nn.A1_p.*(1-nn.A1_p);
elseif strcmp(nn.fo,'linear')
    deri=1;
elseif strcmp(nn.fo,'tanh_opt')
    deri=1 - nn.A1_p.^2;
end
d_bri2{1}=-(nn.A1-nn.A1_p).*deri;

if strcmp(nn.fb,'sigm')
    deri=nn.Y2.*(1-nn.Y2);
elseif strcmp(nn.fb,'linear')
    deri=1;
elseif strcmp(nn.fb,'tanh_opt')
    deri=1 - nn.Y2.^2;
end
d_bri2{2}=(d_bri2{1}*nn.H2{2}').*deri;

if strcmp(nn.f,'sigm')
    deri=nn.A2.*(1-nn.A2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A2.^2;
end
d_bri2{3}=(d_bri2{2}*nn.H2{1}').*deri;

dP2_1{1}=batch_corr_x2'*d_bri2{3}/m;
%2.
d_bri1_ano=(nn.A2-nn.A2_p).*deri;
dP2_1{2}=batch_corr_x2'*d_bri1_ano/m;
%3.
if strcmp(nn.fo,'sigm')
    deri=nn.A1_p.*(1-nn.A1_p);
elseif strcmp(nn.fo,'linear')
    deri=1;
elseif strcmp(nn.fo,'tanh_opt')
    deri=1 - nn.A1_p.^2;
end
d_re1{3}=(d_re1{1}*nn.P1{2}').*deri;

if strcmp(nn.fb,'sigm')
    deri=nn.Y2.*(1-nn.Y2);
elseif strcmp(nn.fb,'linear')
    deri=1;
elseif strcmp(nn.fb,'tanh_opt')
    deri=1 - nn.Y2.^2;
end
d_re1{4}=(d_re1{3}*nn.H2{2}').*deri;

if strcmp(nn.f,'sigm')
    deri=nn.A2.*(1-nn.A2);
elseif strcmp(nn.f,'linear')
    deri=1;
elseif strcmp(nn.f,'tanh_opt')
    deri=1 - nn.A2.^2;
end
d_re1{5}=(d_re1{4}*nn.H2{1}').*deri;
dP2_1{3}=batch_corr_x2'*d_re1{5}/m;

dP2{1}=dP2_1{1}+dP2_1{2}+dP2_1{3};

%H1{2}
%1.
dH1_2{1}=nn.Y1'*d_bri1{1}/m;
%2.
dH1_2{2}=nn.Y1'*d_re2{3}/m;

dH1{2}=dH1_2{1}+dH1_2{2};

%H1{1}
%1.
dH1_1{1}=nn.A1'*d_bri1{2}/m;
%2.
dH1_1{2}=nn.A1'*d_re2{4}/m;

dH1{1}=dH1_1{1}+dH1_1{2};

%H2{2}
%1.
dH2_2{1}=nn.Y2'*d_bri2{1}/m;
%2.
dH2_2{2}=nn.Y2'*d_re1{3}/m;

dH2{2}=dH2_2{1}+dH2_2{2};

%H2{1}
%1.
dH2_1{1}=nn.A2'*d_bri2{2}/m;
%2.
dH2_1{2}=nn.A2'*d_re1{4}/m;

dH2{1}=dH2_1{1}+dH2_1{2};

%BO1
dBO1=sum(d_re1{1})/m;

%BI2
%1.
dBI2_1{1}=sum(d_re2{2})/m;
%2.
dBI2_1{2}=sum(d_equ2)/m;

dBI2=dBI2_1{1}+dBI2_1{2};

%BA1
dBA1_1{1}=sum(d_bri1{3})/m;
dBA1_1{2}=sum(d_bri2{1})/m;%bridge
dBA1_1{3}=sum(d_bri2_ano)/m;
dBA1_1{4}=sum(d_re2{5})/m;
dBA1_1{5}=sum(d_re1{3})/m;

dBA1=dBA1_1{2}+dBA1_1{1}+dBA1_1{3}+dBA1_1{4}+dBA1_1{5};

%BO2
dBO2=sum(d_re2{1})/m;

%BI1
%1.
dBI1_1{1}=sum(d_re1{2})/m;
%2.
dBI1_1{2}=sum(d_equ1)/m;

dBI1=dBI1_1{1}+dBI1_1{2};

%BA2
dBA2_2{1}=sum(d_bri2{3})/m;
dBA2_2{2}=sum(d_bri1{1})/m;
dBA2_2{3}=sum(d_bri1_ano)/m;
dBA2_2{4}=sum(d_re1{5})/m;
dBA2_2{5}=sum(d_re2{3})/m;

dBA2=dBA2_2{2}+dBA2_2{1}+dBA2_2{3}+dBA2_2{4}+dBA2_2{5};

%B1
dB1_1{1}=sum(d_bri1{2})/m;
dB1_1{2}=sum(d_re2{4})/m;

dB1=dB1_1{1}+dB1_1{2};

%B2
dB2_1{1}=sum(d_bri2{2})/m;
dB2_1{2}=sum(d_re1{4})/m;

dB2=dB2_1{1}+dB2_1{2};

%%%% 2-step update
for i=1:2
    %L1   
    W = nn.K1{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dK1{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vK1_2{i} = momentum2*vK1_2{i} + dW;
        dW=vK1_2{i};
    end
             
    nn.K1{i} = nn.K1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.P1{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dP1{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vP1_2{i} = momentum2*vP1_2{i} + dW;
        dW=vP1_2{i};
    end
             
    nn.P1{i} = nn.P1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.K2{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dK2{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vK2_2{i} = momentum2*vK2_2{i} + dW;
        dW=vK2_2{i};
    end
             
    nn.K2{i} = nn.K2{i} - dW;
end

for i=1:2
    %L1   
    W = nn.P2{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dP2{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vP2_2{i} = momentum2*vP2_2{i} + dW;
        dW=vP2_2{i};
    end
             
    nn.P2{i} = nn.P2{i} - dW;
end

for i=1:2
    %L1   
    W = nn.H1{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dH1{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vH1{i} = momentum2*vH1{i} + dW;
        dW=vH1{i};
    end
             
    nn.H1{i} = nn.H1{i} - dW;
end

for i=1:2
    %L1   
    W = nn.H2{i};   
    L1=W./sqrt(W.^2+0.000001);         
    
    dW = dH2{i} + wL1 * L1 + wL2 * W;
    dW = nn.alpha * dW;
    
    if(momentum2>0)
        vH2{i} = momentum2*vH2{i} + dW;
        dW=vH2{i};
    end
             
    nn.H2{i} = nn.H2{i} - dW;
end

db = nn.alpha * dBO1;
if(momentum2>0)
    vBO1_2= momentum2*vBO1_2 + db;
    db=vBO1_2;
end
nn.BO1=nn.BO1-db;

db = nn.alpha * dBO2;
if(momentum2>0)
    vBO2_2= momentum2*vBO2_2 + db;
    db=vBO2_2;
end
nn.BO2=nn.BO2-db;

db = nn.alpha * dBI1;
if(momentum2>0)
    vBI1_2= momentum2*vBI1_2 + db;
    db=vBI1_2;
end
nn.BI1=nn.BI1-db;

db = nn.alpha * dBI2;
if(momentum2>0)
    vBI2_2= momentum2*vBI2_2 + db;
    db=vBI2_2;
end
nn.BI2=nn.BI2-db;

db = nn.alpha * dBA1;
if(momentum2>0)
    vBA1_2= momentum2*vBA1_2 + db;
    db=vBA1_2;
end
nn.BA1=nn.BA1-db;

db = nn.alpha * dBA2;
if(momentum2>0)
    vBA2_2= momentum2*vBA2_2 + db;
    db=vBA2_2;
end
nn.BA2=nn.BA2-db;

db = nn.alpha * dB1;
if(momentum2>0)
    vB1= momentum2*vB1 + db;
    db=vB1;
end
nn.B1=nn.B1-db;

db = nn.alpha * dB2;
if(momentum2>0)
    vB2= momentum2*vB2 + db;
    db=vB2;
end
nn.B2=nn.B2-db;

%%%%%%%%%%%%%%%%%%%% 2-step is over %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear dK1 dK2 dP2 dP1 dBA2 dBI2 dBO2 dBO1 dBI1 dBA1 dH1 dH2 dB1 dB2 dBN1 dBN2 dQ1 dQ2
clear db W dW L1
end

%if(j>250)
%    nn.alpha=0.0005;
%    nn.alpha_ae=0.0005;
%end

nn=nnff(nn,x1,x2,x1,x2);

L_A(j)=nn.Lc{3};
L_I(j)=nn.Lc{7};
str_train_error = ['epoch ' num2str(j) '/' num2str(maxEpoch) '.1-step train error: ' num2str(nn.Lc{1}) ' and '...
    num2str(nn.Lc{2}) '.2-step train error: ' num2str(nn.Lc{3}) ','...
    num2str(nn.Lc{4}) ' and ' num2str(nn.Lc{5}) ',' num2str(nn.Lc{6}) ' and ' num2str(nn.Lc{7})];

fprintf(fid,'%s\n',str_train_error);
disp(str_train_error);

end

K1=nn.K1;P1=nn.P1;
K2=nn.K2;P2=nn.P2;
BI1=nn.BI1;BA1=nn.BA1;1;BO1=nn.BO1;
BI2=nn.BI2;BA2=nn.BA2;BO2=nn.BO2;
B1=nn.B1;B2=nn.B2;
H1=nn.H1;H2=nn.H2;
Q1=nn.Q1;Q2=nn.Q2;
BN1=nn.BN1;BN2=nn.BN2;
save Weights K1 P1 K2 P2 BI1 BI2 BA1 BA2 BO1 BO2 H1 H2 B1 B2 Q1 Q2 BN1 BN2;
