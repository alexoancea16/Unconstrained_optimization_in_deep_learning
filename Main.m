% Unconstrained optimization in Deep Learning

clc; clear; close all;

%% Data selection
load date.mat;
N = 160;      % Number of examples for training
n = 11;       % Number of features
m = 12;       % Number of neurons for the input layer, and also for the hidden layer
A(1:160,12) = 1;      % Training matrix
T(1:40,12) = 1;       % Test matrix