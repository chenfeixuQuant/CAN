function nn=nnff(nn,batch_corr_x1,batch_corr_x2,batch_x1,batch_x2)

m=size(batch_corr_x1,1);

%% activation value
%I1:mxd
if strcmp(nn.f,'sigm')
    nn.I1=sigmoid(batch_corr_x1*nn.K1{1}+repmat(nn.BI1,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.I1=batch_corr_x1*nn.K1{1}+repmat(nn.BI1,[m 1]);
elseif strcmp(nn.f,'tanh_opt')
    nn.I1=tanh_opt(batch_corr_x1*nn.K1{1}+repmat(nn.BI1,[m 1]));
end
if(nn.sparsityPenaltyI>0)  
    nn.sI1 = 0.99 * nn.sI1 + 0.01 * mean(nn.I1, 1);
end

%A1:mxp
if strcmp(nn.f,'sigm')
    nn.A1=sigmoid(batch_corr_x1*nn.P1{1}+repmat(nn.BA1,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.A1=batch_corr_x1*nn.P1{1}+repmat(nn.BA1,[m 1]);
elseif strcmp(nn.f,'tanh_opt')
    nn.A1=tanh_opt(batch_corr_x1*nn.P1{1}+repmat(nn.BA1,[m 1]));
end
if(nn.sparsityPenaltyA>0)  
    nn.sA1 = 0.99 * nn.sA1 + 0.01 * mean(nn.A1, 1);
end

%N1:mxp
if strcmp(nn.f,'sigm')
    nn.N1=sigmoid(batch_corr_x1*nn.Q1{1}+repmat(nn.BN1,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.N1=batch_corr_x1*nn.Q1{1}+repmat(nn.BN1,[m 1]);
elseif strcmp(nn.f,'tanh_opt')
    nn.N1=tanh_opt(batch_corr_x1*nn.Q1{1}+repmat(nn.BN1,[m 1]));
end
if(nn.sparsityPenaltyN>0)  
    nn.sN1 = 0.99 * nn.sN1 + 0.01 * mean(nn.N1, 1);
end

%batch_corr_x1~
if strcmp(nn.s,'sigm')
    nn.batch_x1_p=sigmoid(nn.I1*nn.K1{2}+nn.A1*nn.P1{2}+nn.N1*nn.Q1{2}+repmat(nn.BO1,[m 1]));
elseif strcmp(nn.s,'linear')
    nn.batch_x1_p=nn.I1*nn.K1{2}+nn.A1*nn.P1{2}+nn.N1*nn.Q1{2}+repmat(nn.BO1,[m 1]);
elseif strcmp(nn.s,'tanh_opt')
    nn.batch_x1_p=tanh_opt(nn.I1*nn.K1{2}+nn.A1*nn.P1{2}+nn.N1*nn.Q1{2}+repmat(nn.BO1,[m 1]));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%I2:mxd
if strcmp(nn.f,'sigm')
    nn.I2=sigmoid(batch_corr_x2*nn.K2{1}+repmat(nn.BI2,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.I2=batch_corr_x2*nn.K2{1}+repmat(nn.BI2,[m 1]);
elseif strcmp(nn.f,'tanh_opt')
    nn.I2=tanh_opt(batch_corr_x2*nn.K2{1}+repmat(nn.BI2,[m 1]));
end
if(nn.sparsityPenaltyI>0)  
    nn.sI2 = 0.99 * nn.sI2 + 0.01 * mean(nn.I2, 1);
end

%A2:mxp
if strcmp(nn.f,'sigm')
    nn.A2=sigmoid(batch_corr_x2*nn.P2{1}+repmat(nn.BA2,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.A2=batch_corr_x2*nn.P2{1}+repmat(nn.BA2,[m 1]);
elseif strcmp(nn.f,'tanh_opt')
    nn.A2=tanh_opt(batch_corr_x2*nn.P2{1}+repmat(nn.BA2,[m 1]));
end
if(nn.sparsityPenaltyA>0)  
    nn.sA2 = 0.99 * nn.sA2 + 0.01 * mean(nn.A2, 1);
end

%N2:mxp
if strcmp(nn.f,'sigm')
    nn.N2=sigmoid(batch_corr_x2*nn.Q2{1}+repmat(nn.BN2,[m 1]));
elseif strcmp(nn.f,'linear')
    nn.N2=batch_corr_x2*nn.Q2{1}+repmat(nn.BN2,[m 1]);
elseif strcmp(nn.f,'tanh_opt') 
    nn.N2=tanh_opt(batch_corr_x2*nn.Q2{1}+repmat(nn.BN2,[m 1]));
end
if(nn.sparsityPenaltyN>0)  
    nn.sN2 = 0.99 * nn.sN2 + 0.01 * mean(nn.N2, 1);
end

%batch_corr_x2~
if strcmp(nn.s,'sigm')
    nn.batch_x2_p=sigmoid(nn.I2*nn.K2{2}+nn.A2*nn.P2{2}+nn.N2*nn.Q2{2}+repmat(nn.BO2,[m 1]));
elseif strcmp(nn.s,'linear')
    nn.batch_x2_p=nn.I2*nn.K2{2}+nn.A2*nn.P2{2}+nn.N2*nn.Q2{2}+repmat(nn.BO2,[m 1]);
elseif strcmp(nn.s,'tanh_opt')
    nn.batch_x2_p=tanh_opt(nn.I2*nn.K2{2}+nn.A2*nn.P2{2}+nn.N2*nn.Q2{2}+repmat(nn.BO2,[m 1]));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Y1:fb(A1*H{1}+B)
if strcmp(nn.fb,'sigm')
    nn.Y1=sigmoid(nn.A1*nn.H1{1}+repmat(nn.B1,[m 1]));
elseif strcmp(nn.fb,'linear')
    nn.Y1=nn.A1*nn.H1{1}+repmat(nn.B1,[m 1]);
elseif strcmp(nn.fb,'tanh_opt')
    nn.Y1=tanh_opt(nn.A1*nn.H1{1}+repmat(nn.B1,[m 1]));
end

%Y2:fb(A1*H{1}+B)
if strcmp(nn.fb,'sigm')
    nn.Y2=sigmoid(nn.A2*nn.H2{1}+repmat(nn.B2,[m 1]));
elseif strcmp(nn.fb,'linear')
    nn.Y2=nn.A2*nn.H2{1}+repmat(nn.B2,[m 1]);
elseif strcmp(nn.fb,'tanh_opt')
    nn.Y2=tanh_opt(nn.A2*nn.H2{1}+repmat(nn.B2,[m 1]));
end

%A2~ = fo(Y*H{2}+B2a)
if strcmp(nn.fo,'sigm')
    nn.A2_p=sigmoid(nn.Y1*nn.H1{2}+repmat(nn.BA2,[m 1]));
elseif strcmp(nn.fo,'linear')
    nn.A2_p=nn.Y1*nn.H1{2}+repmat(nn.BA2,[m 1]);
elseif strcmp(nn.fo,'tanh_opt')
    nn.A2_p=tanh_opt(nn.Y1*nn.H1{2}+repmat(nn.BA2,[m 1]));
end

%A1~ = fo(Y*H{2}+B2a)
if strcmp(nn.fo,'sigm')
    nn.A1_p=sigmoid(nn.Y2*nn.H2{2}+repmat(nn.BA1,[m 1]));
elseif strcmp(nn.fo,'linear')
    nn.A1_p=nn.Y2*nn.H2{2}+repmat(nn.BA1,[m 1]);
elseif strcmp(nn.fo,'tanh_opt')
    nn.A1_p=tanh_opt(nn.Y2*nn.H2{2}+repmat(nn.BA1,[m 1]));
end

%batch_corr_x2' = s(A2~*P2{2}+I2*K2{2}+E2*Q2{2}+B2)
if strcmp(nn.s,'sigm')
    nn.batch_x2_p2=sigmoid(nn.A2_p*nn.P2{2}+nn.I2*nn.K2{2}+repmat(nn.BO2,[m 1]));
elseif strcmp(nn.s,'linear')
    nn.batch_x2_p2=nn.A2_p*nn.P2{2}+nn.I2*nn.K2{2}+repmat(nn.BO2,[m 1]);
elseif strcmp(nn.s,'tanh_opt')
    nn.batch_x2_p2=tanh_opt(nn.A2_p*nn.P2{2}+nn.I2*nn.K2{2}+repmat(nn.BO2,[m 1]));
end

%batch_corr_x1' = s(A2~*P2{2}+I2*K2{2}+E2*Q2{2}+B2)
if strcmp(nn.s,'sigm')
    nn.batch_x1_p2=sigmoid(nn.A1_p*nn.P1{2}+nn.I1*nn.K1{2}+repmat(nn.BO1,[m 1]));
elseif strcmp(nn.s,'linear')
    nn.batch_x1_p2=nn.A1_p*nn.P1{2}+nn.I1*nn.K1{2}+repmat(nn.BO1,[m 1]);
elseif strcmp(nn.s,'tanh_opt')
    nn.batch_x1_p2=tanh_opt(nn.A1_p*nn.P1{2}+nn.I1*nn.K1{2}+repmat(nn.BO1,[m 1]));
end

%% Loss Compute
nn.Lc=cell(1,7);
nn.Lc{1}=1/2*sum(sum((batch_x1-nn.batch_x1_p).^2,2))/m;
nn.Lc{2}=1/2*sum(sum((batch_x2-nn.batch_x2_p).^2,2))/m;

nn.Lc{3}=1/2*sum(sum((nn.A2-nn.A2_p).^2,2))/m;
nn.Lc{4}=1/2*sum(sum((nn.A1-nn.A1_p).^2,2))/m;
nn.Lc{5}=1/2*sum(sum((batch_x2-nn.batch_x2_p2).^2,2))/m;
nn.Lc{6}=1/2*sum(sum((batch_x1-nn.batch_x1_p2).^2,2))/m;
nn.Lc{7}=1/2*sum(sum((nn.I2-nn.I1).^2,2))/m;

nn.L=nn.Lc{1}+nn.Lc{2}+nn.Lc{3}+nn.Lc{4}+nn.Lc{5}+nn.Lc{6}+nn.Lc{7};

end