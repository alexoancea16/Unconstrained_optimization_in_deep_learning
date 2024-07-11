% Unconstrained optimization in Deep Learning

clc; clear; close all;

%% Data selection
load date.mat;
N = 160;      % Number of examples for training
Nt = 40;      % Number of examples for test 
n = 11;       % Number of features
m = 12;       % Number of neurons for the input layer, and also for the hidden layer
A(1:160,12) = 1;      % Training matrix
T(1:40,12) = 1;       % Test matrix
X = rand(m,m);        % Network parameters
x = rand(m,1);
%% Data normalization
for i = 1:n
    A(:,i) = normalization(A(:,i),N);
    T(:,i) = normalization(T(:,i),Nt);
end
e = normalization(e,N);
e_test = normalization(e_test,Nt);

%% Numerical optimization methods

% Gradient Method 
[X_calculat, x_calculat, iteration] = GradientMethod(A, e, X, x);
