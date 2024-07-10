% Derivata functiei de activare 

function der = g_der(z)

der = -(1 +2.*exp(z)).^(-1) .* exp(z);

end