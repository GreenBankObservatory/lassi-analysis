function [] = test_python()
disp("test_python")
% Generate data points with noise
num_points = 5
Tx = linspace(5., 8., num_points)
Ty = Tx;

%rand(1, num_points)
%exp(rand(1, num_points))
%a = exp(2*rand(1, num_points).^2)
%b = (0.5-rand(1,num_points))
%size(a)
%size(b)
%a*b
%(0.5-rand(1,num_points))*exp(2*rand(1, num_points).^2)

%tX = 11.86*cos(2*pi/0.81*Tx-1.32) + 0.64*Tx+4*((0.5-rand(1,num_points))*exp(2*rand(1, num_points).^2))
tX1 = 11.86*cos(2*pi/0.81*Tx-1.32)
%tX2 = 0.64*Tx+4*((0.5-rand(1,num_points))*exp(2*rand(1, num_points).^2))
tX2_1 = 0.64*Tx
tX2_2 = 4*times((0.5-rand(1,num_points)), exp(2*rand(1, num_points).^2))
tX2 = tX2_1 + tX2_2
tX = tX1 + tX2
% tY = -32.14*np.cos(2*np.pi/0.8*Ty-1.94) + 0.15*Ty+7*((0.5-rand(num_points))*np.exp(2*rand(num_points)**2))

% Fit the first set
% Target Function
fitfunc = @(p, x) p(1)*cos(2*pi/p(2)*x+p(3)) + p(4)*x;

% Distance to Target func
errfunc = @(p, y) fitfunc(p(1), p(2)) - y;
% Initial guess for the parameters
p0 = [-15. 0.8 0. -1.];
%p1, success = optimize.leastsq(errfunc, p0[:], args=(Tx, tX))
options = optimset('MaxIter',1E4,'MaxFunEvals',512*512*3*200,'TolFun',1E-10,'TolX',1E-10);
%FIT = lsqcurvefit(errfunc, p0, tX, zeros([ 1 max(size(Tx))]),[],[],options);
FIT = lsqcurvefit(errfunc, p0, tX, [], [], [], options);
disp(FIT)

end
