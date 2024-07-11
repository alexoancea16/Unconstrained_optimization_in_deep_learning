% Gradient Method 
function [X, x, k] = GradientMethod(A, e, X, x)

k = 0;
alpha = 0.001;
k_max = 1000;
eps = 0.01;
N = size(A,1);
M = g(A*X)*x;    % Predictions
err = M - e;
grd_x = (1/N) * (g(A*X))' * err;               % First order derivative of the Loss function for x
grd_X = (1/N) * (g_der(A*X))' * (err * x');    % First order derivative of the Loss function for X
while norm(grd_X) > eps || norm(grd_x) > eps || k < k_max
    X = X - alpha * grd_X;
    x = x - alpha * grd_x;
    M = g(A*X)*x;    
    err = M - e;
    grd_x = (1/N) * (g(A*X))' * err;              
    grd_X = (1/N) * (g_der(A*X))' * (err * x');   
    k = k + 1; 
end

end