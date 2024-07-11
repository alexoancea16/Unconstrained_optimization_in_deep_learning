% Gradient Method 
function [X, x, k, X_evolution, x_evolution, time] = GradientMethod(A, e, X, x)

k = 0;
k_max = 10000;
X_evolution = zeros(k_max, 1);  
x_evolution = zeros(k_max, 1);
time = zeros(k_max, 1);
alpha = 0.001;
eps = 0.0001;
N = size(A, 1);
M = g(A*X)*x;    % Predictions
err = M - e;
grd_x = (1/N) * (g(A*X))' * err;               % First order derivative of the Loss function for x
grd_X = (1/N) * (g_der(A*X))' * (err * x');    % First order derivative of the Loss function for X
tic
while norm(grd_X) > eps && norm(grd_x) > eps && k < k_max
    X = X - alpha * grd_X;
    x = x - alpha * grd_x;
    M = g(A*X)*x;    
    err = M - e;
    grd_x = (1/N) * (g(A*X))' * err;              
    grd_X = (1/N) * (g_der(A*X))' * (err * x');   
    k = k + 1; 
    X_evolution(k) = norm(grd_X);
    x_evolution(k) = norm(grd_x);
    time(k) = toc;
end

end