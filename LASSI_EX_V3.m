function [ az_loc,el_loc, r_sm ] = LASSI_EX_V3( L,n )

% L - x, y, z matrix of data (originally from csv file
% n - unused

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Rx is a function of theta - returns this matrix
Rx = @(theta) [ 1 0 0; 0 cos(theta) -sin(theta); 0 sin(theta) cos(theta)];
Ry = @(theta) [ cos(theta) 0 sin(theta);0 1 0; -sin(theta) 0 cos(theta)];

% don't need one for Rz
%Rz = @(theta) [ cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
Rz = @(theta) [ 1 0 0; 0 1 0; 0 0 1];

% PRY - Pitch Roll Yaw - 3D airplane 
% We are going to fit for angles to find the best fit for a tilted parabaloid
% tilt defined by PRY
Prime = @(PRY) Rz(PRY(1))*Ry(PRY(2))*Rx(PRY(3));
Prime_x = @(PRY,V) [1 0 0]*Prime(PRY)*V;
Prime_y = @(PRY,V) [0 1 0]*Prime(PRY)*V;
Prime_z = @(PRY,V) [0 0 1]*Prime(PRY)*V;

%f = @(b, x)((x(:,1)-b(1)).^2 * b(2) + (x(:,2)-b(3)).^2 * b(4) ...
    %+( x(:,1)-b(1) ) * b(5) + ( x(:,2)-b(3) ) * b(6) - b(7));
% f = @(b,x) Prime_z(b(1:3),x)...
%     -b(4)*( (Prime_x(b(1:3),x)- b(5)).^2 ...
%     +(Prime_y(b(1:3),x)-b(6)).^2 ) - b(7);

% GBT MEMO 155: A Summary of the GBT Optics Design

% b is everything we are going to fit for:
% b(4) = focal length (f in eq.)
% b(1:3) = angles
% b(7) = z translation = z0
% b(5) = x translation
% b(6) = y translation
% that's the minimation problem: we want this function f to be zero!
% used for fitting
% here x arg is the full XYZ matrix (or vector list)
f = @(b,x) 4 * b(4) * ( b(4)-(Prime_z(b(1:3),x)-b(7) )  )...
    -( (Prime_x(b(1:3),x)- b(5)).^2 ...
    +(Prime_y(b(1:3),x)-b(6)).^2 );

fn = @(b, x) f1(b, x) - f2(b, x)
f1 = @(b, x)  4 * b(4) * ( b(4)-(Prime_z(b(1:3),x)-b(7) )  )
f2 = @(b, x) ( (Prime_x(b(1:3),x)- b(5)).^2 + (Prime_y(b(1:3),x)-b(6)).^2 )

% fz = @(b,x) b(4)*( (Prime_x(b(1:3),x)- b(5)).^2 ...
%     +(Prime_y(b(1:3),x)-b(6)).^2 ) + b(7);

% returns the z after the fit from above
fz = @(b,x) b(4) - 0.25/b(4)*( (Prime_x(b(1:3),x)- b(5)).^2 ...
    +(Prime_y(b(1:3),x)-b(6)).^2 );

% extract original data 
x=L(:,1); y=L(:,2); z=L(:,3);
% pop out the NANs
xy=L; xy(isnan(xy(:,1)),:)=[]; xy(isnan(xy(:,2)),:)=[]; xy(isnan(xy(:,3)),:)=[];

% options:
% MaxIter : 1E4
% MaxFunEvals : numel(L) 512*512*3
% TolFun : 1E-10
% TolX : 1E-10
options = optimset('MaxIter',1E4,'MaxFunEvals',numel(L),'TolFun',1E-10,'TolX',1E-10);
%options = optimset('MaxIter',1E4,'MaxFunEvals',1E4,'TolFun',1E-10,'TolX',1E-10);

% we are fitting to our function 'f' above, and our initial seed value is 
% [0 0 0 - angles
% 60 - focal length (from memo)
% translations are minimum values ]
% ' in matlab is 'transposed'; xy' is our data transposed
FIT = lsqcurvefit(f, [ 0 0 0 60 min(x) min(y) min(z) ], xy', ...
    zeros([ 1 max(size(xy))]),[],[],options);

disp("FIT: ")
disp(FIT)
disp(size(FIT))

% so lsqcurvefit gave us values for 'b' from above
Focus=FIT(4);
% coordinate transform of our data to the bore-sight along z frame
X_prime= Prime_x(FIT(1:3),L');
Y_prime= Prime_y(FIT(1:3),L');
Z_prime= Prime_z(FIT(1:3),L')-FIT(end);


% it's not enough just to translate them like above,
% you also have to remove the fits themselves!  
% only then do we see the bumps
% why do we need to do this? subtract both parabola fits? 
Z_fit=fz(FIT,L');
Delta=(Z_prime-Z_fit);

disp("Delta:")
disp(size(Delta))
% do the above for the ref scan, and the bump scan,
% and subtract the Detla above for both, and you'll see bumps baby!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% everything below is smoothing that we don't need.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%V_prime(:,1)= X_prime; V_prime(:,2)= Y_prime; V_prime(:,3)= Z_prime;
%V=(inv(Prime(FIT(1:3)))*V_prime')';

%Delta=z-f(FIT,xy);
%[az,el,r] = cart2sph(x,y,z);

%[az,el,r] = cart2sph(X_prime,Y_prime,Z_prime);
%az(az<0)=az(az<0)+2*pi;


%d_az=(max(az)-min(az))/(n-1);
%d_el=(max(el)-min(el))/(n-1);
%az_range=min(az):d_az:max(az);
%el_range=min(el):d_el:max(el);

%[az_loc,el_loc] = meshgrid(az_range,el_range);
%sig_el=d_el/2;
%sig_az=d_az/2;
%sig_el=0.003;
%sig_az=0.003;
%for j=1:n
%    for k=1:n
%        w=2*pi*exp( (-cos(el_loc(j,k)).^2.* (az - az_loc(j,k)).^2 ./( 2.*sig_az^2 )...
%            -(el-el_loc(j,k)).^2 ./(2.*sig_el^2 )) );
%        Norm=sum(w);
%        if Norm==0;
%            Norm=1;
%            r_sm(j,k)=min( r ); 
%        else
%            w=w./Norm;
%            r_sm(j,k)=sum( r .* w);
%        end
%    end
%end
%figure;
%subplot(3,1,1)
%    imagesc(r_sm)
%    axis image
%    axis off
%subplot(3,1,[2,3])
%surf(az_loc,el_loc,r_sm)
%hold on
%plot3(az,el,r,'.');
%shading interp
%axis tight
%

%[sm_x,sm_y,sm_z] = sph2cart(az_loc,el_loc,r_sm);
%sm_xy=[reshape(sm_x,numel(sm_x),1),reshape(sm_y,numel(sm_y),1)];
%sm_z_fit=reshape(f(FIT,sm_xy),size(sm_z));

%figure;
%Delta=sm_z-sm_z_fit;
%Delta(abs(Delta)>.2)=0;
%surf(sm_x,sm_y,Delta);
%axis tight
%shading interp

disp("LASSI_EX_V3 done");

end

%% 
