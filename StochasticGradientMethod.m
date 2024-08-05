% Stochastic Gradient Method
function [X, x, k, X_evolution, x_evolution, time] = StochasticGradientMethod(A, e, X, x, eps, alpha, k_max)

k = 0;
X_evolution = zeros(k_max, 1);  
x_evolution = zeros(k_max, 1);
time = zeros(k_max, 1);
N = size(A, 1);
M = g(A*X)*x;    % Predictions
err = M - e;
grd_x = (1/N) * (g(A*X))' * err;               % First order derivative of the Loss function for x
grd_X = (1/N) * (g_der(A*X))' * (err * x');    % First order derivative of the Loss function for X
tic
while norm(grd_X) > eps && norm(grd_x) > eps && k < k_max
    for i = 1:N
        M_i = g(A(i,:) * X) * x;
        err_i = M_i - e(i);
        grd_x_i = g(A(i,:) * X)' * err_i;                
        grd_X_i = g_der(A(i,:) * X)' * (err_i * x');     
        X = X - alpha * grd_X_i;
        x = x - alpha * grd_x_i;
    end
    M = g(A*X) * x;    
    err = M - e;
    grd_x = (1/N) * (g(A*X))' * err;              
    grd_X = (1/N) * (g_der(A*X))' * (err * x'); 
    k = k + 1;
    X_evolution(k) = norm(grd_X);
    x_evolution(k) = norm(grd_x); 
    time(k) = toc;
end

end