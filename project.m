% %%%%%%%%%%%%-----------------project-----------

%%%%%%%%%%%%%%%%%%% This code will convert all the categorical data to
%%%%%%%%%%%%%%%%%%% numerical data
clear all; close all;clc;
%% [num,txt,raw] = xlsread(___) additionally returns the text fields in cell array txt, 
%% and both numeric and text data in cell array raw
[n,t,r]=xlsread('data.csv');
[R,C]=size(r(:,:));
%%%%consider from the row the raw data
n_1=n(2:R,:);t_1=t(2:R,:);r_1=r(2:R,:);
A=cat2num(r_1);

filename = 'proj_tr_data.xlsx';



% 
xlswrite(filename,A);

clear all;close all;clc;
[n,t,r]=xlsread('quiz.csv');
[R,C]=size(r(:,:));
%n_1=n(2:R,:);
%t_1=t(2:R,:);
r_1=r(2:R,:);
A=cat2num(r_1);

filename = 'proj_test_data.xlsx';




xlswrite(filename,A)
