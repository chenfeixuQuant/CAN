%对测试集进行PCA+LDA降维
clear all;clc
load test_fg
load ldaMapping
load pcaMapping
load Weights3000

test_label = test_source_fg(:,end-1);
age_source_label = test_source_fg(:,end);
age_target_label = test_target_fg(:,end);

x1_test = test_source_fg(:,1:end-2)/255;
x2_test = test_target_fg(:,1:end-2)/255;

x1_mean = mean(x1_test,1);
x1_test = (x1_test - repmat(x1_mean,size(x1_test,1),1));

x2_mean = mean(x2_test,1);
x2_test = (x2_test - repmat(x2_mean,size(x2_test,1),1));

m=size(x1_test,1);
probe = sigmoid(x1_test*K1{1}+repmat(BI1,[m 1]));
gallery = sigmoid(x2_test*K2{1}+repmat(BI2,[m 1]));

probe=bsxfun(@minus,probe,pca_mapping.mean)*pca_mapping.M;
gallery=bsxfun(@minus,gallery,pca_mapping.mean)*pca_mapping.M;

probe=bsxfun(@minus,probe,lda_mapping.mean)*lda_mapping.M;
gallery=bsxfun(@minus,gallery,lda_mapping.mean)*lda_mapping.M;

save features probe gallery 
save test_labels test_label age_source_label age_target_label

