% Gradient Method 
function [X,x,k] = GradientMethod(A,e,X,x)

k = 0;
alpha = 0.001;
k_max = 1000;
eps = 0.01;
N = size(A,1);
tic
while k < k_max
    k = k + 1;
    M = g(A*X)*x;  % predictions
    err = M - e;
    grd_x = (1/N) * g(A*X).' * err;
    grd_X = (1/N) * g_der(A*X).' * (err * x.');
    X = X - alpha * grd_X;
    x = x - alpha * grd_x;
    if norm(grd_X) <= eps || norm(grd_x) <= eps
            break;
    end
end
end