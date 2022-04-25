function mpc_controller = get_pendulum_mpc()

g = 10; % gravitational coefficient
m = 0.15; % mass
l = 0.5; % length
mu = 0.05; % frictional coefficient
dt = 0.02; % sampling period

AG = [1,      dt;...
    g/l*dt, 1-mu/(m*l^2)*dt];
BG = [0; dt/(m*l^2)];

nG = size(AG, 1);
nu = size(BG, 2);

% Linear discrete-time prediction model
model=LTISystem('A', AG, 'B', BG);

% Input constraints
model.u.min = -2; model.u.max = 2;

% State constraints
model.x.min = [-2.5; -6];
model.x.max = [2.5; 6];

% constraint sets represented as polyhedra
X = Polyhedron('lb',model.x.min,'ub',model.x.max);
U = Polyhedron('lb',model.u.min,'ub',model.u.max);

% Penalties in the cost function
Q = eye(nG);
R = 10;
model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);

% Maximal Invariant Set Computation
[Pinf,Kinf,L] = idare(AG,BG,Q,R);% closed loop system
Acl=AG-BG*Kinf;
S=X.intersect(Polyhedron('H',[-U.H(:,1:nu)*Kinf U.H(:,nu+1)]));
Oinf=max_pos_inv(Acl,S);

model.x.with('terminalSet');
model.x.terminalSet = Oinf;
model.x.with('terminalPenalty');
model.x.terminalPenalty = QuadFunction(Pinf);

% Online MPC object
mpc_controller = MPCController( model, 6 );
end