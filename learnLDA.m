%LDA learning
%identity extraction
clear all;clc
load Weights3000
load train_set5000
load pcaMapping

train_label1=train_source(:,end-1);
train_label2=train_label1;

x1 = train_source(:,1:end-2)/255;
x2 = train_target(:,1:end-2)/255;

x1_mean = mean(x1,1);
x1 = (x1 - repmat(x1_mean,size(x1,1),1));

x2_mean = mean(x2,1);
x2 = (x2 - repmat(x2_mean,size(x2,1),1));

%1.PCA
m=size(x1,1);
a = sigmoid(x1*K1{1}+repmat(BI1,[m 1]));
b = sigmoid(x2*K2{1}+repmat(BI2,[m 1]));

a=bsxfun(@minus,a,pca_mapping.mean)*pca_mapping.M;
b=bsxfun(@minus,b,pca_mapping.mean)*pca_mapping.M;

%2.LDA
X=[a;b];
l=[train_label1;train_label2];
[newX, lda_mapping]=lda(X,l,100);

save ldaMapping lda_mapping
save labels train_label1 train_label2
