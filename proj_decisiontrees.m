%%%%%%%%%%%%%%%%decisiontrees----------

clear all; close all;clc;

x=xlsread('proj_tr_data.xlsx');
y=xlsread('proj_test_data.xlsx');

[Noofsamp,Noof_feat]=size(x);
data1=x(:,1:Noof_feat-1);
labels1=x(:,Noof_feat);
tree = ClassificationTree.fit(data1,labels1);
pred = predict(tree,y) ;
%err=length(find(labels1~=pred))/length(labels1);

n=(1:length(pred))';
A=[n pred];

filename = 'proj_test_labels.csv';
csvwrite(filename,{'Id', 'Prediction'});
csvwrite(filename,A,2,0);
