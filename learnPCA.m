%ѧϰPCA�任
clear all;clc
load train_set5000
load Weights3000

x1 = train_source(:,1:end-2)/255;
x2 = train_target(:,1:end-2)/255;

x1_mean = mean(x1,1);
x1 = (x1 - repmat(x1_mean,size(x1,1),1));

x2_mean = mean(x2,1);
x2 = (x2 - repmat(x2_mean,size(x2,1),1));

m=size(x1,1);
a = sigmoid(x1*K1{1}+repmat(BI1,[m 1]));
b = sigmoid(x2*K2{1}+repmat(BI2,[m 1]));

X=[a;b];
[newX, pca_mapping]=pca(X,400);
all=sum(pca_mapping.lambda);
sub=sum(pca_mapping.newlambda);
a=sub/all;

disp(['retain: ' num2str(a)]);

save pcaMapping pca_mapping