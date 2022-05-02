function projection = Project(S, A, B, u, x)
C = S.H;
c = S.b;
u_hat = sdpvar(1,1);
e = u - u_hat;
x_next = A*x + B*u_hat;
F = [];
x_ = [x_next; u_hat];
F = [F; C*x_ <= c];
options = sdpsettings('verbose',0);
optimize(F, e'*e, options);
projection = value(u_hat);
end