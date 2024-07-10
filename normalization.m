% Data normalization - for an array
function x = normalization(x,N)

for i = 1:N
    if x(i) < 0.1
        x(i) = 0.1;
    end
end
for i = 1:N
    x(i) = x(i)/norm(x);
end
for i = 1:N
    if x(i) < 0.1
        x(i) = 0.1;
    end
    if x(i) > 1
        x(i) = 0.9;
end
end