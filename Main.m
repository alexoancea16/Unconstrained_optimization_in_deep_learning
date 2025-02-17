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
alpha = 0.001;
eps = 0.0001;
maxIter = 10000;
%% Data normalization
for i = 1:n
    A(:,i) = normalization(A(:,i),N);
    T(:,i) = normalization(T(:,i),Nt);
end
e = normalization(e,N);
e_test = normalization(e_test,Nt);

%% Numerical optimization methods

% Gradient Method 
[X_GM, x_GM, iteration_GM, X_evolution_GM, x_evolution_GM, time_GM] = GradientMethod(A, e, X, x, eps, alpha, maxIter);
Y = g(T*X_GM)*x_GM;
% Stochastic Gradient Method
[X_SG, x_SG, iteration_SG, X_evolution_SG, x_evolution_SG, time_SG] = StochasticGradientMethod(A, e, X, x, eps, alpha, maxIter);
Z = g(T*X_SG)*x_SG;

%% Comparison of results for the use of optimization methods

% Gradient Method - Diagrames
figure();
figGM_x = semilogy(1:iteration_GM-1,x_evolution_GM(1:iteration_GM-1,1));
figGM_x.LineWidth = 1;
figGM_x.Color = 'r';
title('Gradient Method - The evolution of the x-dependent criterion according to the iterations', 'FontSize', 10, 'FontWeight', 'bold');
xlabel("Number of iterations");
ylabel("The gradient norm in x");
grid on;
figure();
figGM_X = semilogy(1:iteration_GM-1,X_evolution_GM(1:iteration_GM-1,1));
figGM_X.LineWidth = 1;
figGM_X.Color = 'b';
title('Gradient Method - The evolution of the X-dependent criterion according to the iterations', 'FontSize', 10, 'FontWeight', 'bold');
xlabel("Number of iterations");
ylabel("The gradient norm in X");
grid on;
figure();
figGM_t = semilogy(1:iteration_GM-1,time_GM(1:iteration_GM-1,1));
figGM_t.LineWidth = 1;
figGM_t.Color = 'g';
title('Gradient Method - The evolution of the time according to the iterations', 'FontSize', 10, 'FontWeight', 'bold');
xlabel("Time evolution");
ylabel("The gradient norm in X");
grid on;

% Gradient Method - results
% Mean squared error
MSE = (1/Nt)*(norm(Y - e_test))^2;
disp("GM Mean squared error: ");
disp(MSE);
% Mean absolute error
MAE = 0;
for i = 1:Nt
    MAE = MAE + abs(e_test(i) - Y(i));
end
MAE = (1/Nt)*MAE;
disp("GM Mean absolute error: ");
disp(MAE);
% Score
R = 1;
R1 = 0;
R2 = 0;
for i = 1:Nt
    R1 = R1 + (e_test(i) - Y(i))^2;
    R2 = R2 + (e_test(i) - mean(e_test))^2;
end
R = R - R1/R2;
disp("GM Network score: ");
disp(R);

% Stochastic Gradient Method - Diagrames
figure();
figSG_x = semilogy(1:iteration_SG-1,x_evolution_SG(1:iteration_SG-1,1));
figSG_x.LineWidth = 1;
figSG_x.Color = 'r';
title('Stochastic Gradient Method - The evolution of the x-dependent criterion according to the iterations', 'FontSize', 8, 'FontWeight', 'bold');
xlabel("Number of iterations");
ylabel("The gradient norm in x");
grid on;
figure();
figSG_X = semilogy(1:iteration_SG-1,X_evolution_SG(1:iteration_SG-1,1));
figSG_X.LineWidth = 1;
figSG_X.Color = 'b';
title('Stochastic Gradient Method - The evolution of the X-dependent criterion according to the iterations', 'FontSize', 8, 'FontWeight', 'bold');
xlabel("Number of iterations");
ylabel("The gradient norm in X");
grid on;
figure();
figSG_t = semilogy(1:iteration_SG-1,time_SG(1:iteration_SG-1,1));
figSG_t.LineWidth = 1;
figSG_t.Color = 'g';
title('Stochastic Gradient Method - The evolution of the time according to the iterations', 'FontSize', 8, 'FontWeight', 'bold');
xlabel("Time evolution");
ylabel("The gradient norm in X");
grid on;

% Stochastic Gradient Method - results
% Mean squared error
MSE = (1/Nt)*(norm(Z - e_test))^2;
disp("SG Mean squared error: ");
disp(MSE);
% Mean absolute error
MAE = 0;
for i = 1:Nt
    MAE = MAE + abs(e_test(i) - Z(i));
end
MAE = (1/Nt)*MAE;
disp("SG Mean absolute error: ");
disp(MAE);
% Score
R = 1;
R1 = 0;
R2 = 0;
for i = 1:Nt
    R1 = R1 + (e_test(i) - Z(i))^2;
    R2 = R2 + (e_test(i) - mean(e_test))^2;
end
R = R - R1/R2;
disp("SG Network score: ");
disp(R);