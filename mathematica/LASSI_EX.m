function [ az_loc,el_loc, r_sm ] = LASSI_EX( L,n )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
f = @(b, x)((x(:,1)-b(1)).^2 / (b(2)) + (x(:,2)-b(3)).^2 / (b(4)) - b(5));

x=L(:,1); y=L(:,2); z=L(:,3);
xy=L(:,1:2);

FIT = lsqcurvefit(f, [ min(x) 1 min(y) 1 min(z) ], xy, z);

%Delta=z-f(FIT,xy);
[az,el,r] = cart2sph(x,y,z);
az(az<0)=az(az<0)+2*pi;


d_az=(max(az)-min(az))/(n-1);
d_el=(max(el)-min(el))/(n-1);
az_range=min(az):d_az:max(az);
el_range=min(el):d_el:max(el);

[az_loc,el_loc] = meshgrid(az_range,el_range);
%sig_el=d_el/2;
%sig_az=d_az/2;
sig_el=0.001;
sig_az=0.001;
for j=1:n
    for k=1:n
        w=2*pi*exp( (- (az - az_loc(j,k)).^2 ./( 2.*sig_az^2 )-...
            (el-el_loc(j,k)).^2 ./(2.*sig_el^2 )));
        Norm=sum(w);
        if Norm==0;
            Norm=1;
            r_sm(j,k)=min( r ); 
        else
            w=w./Norm;
            r_sm(j,k)=sum( r .* w);
        end
    end
end
figure;
subplot(3,1,1)
    imagesc(r_sm)
    axis image
    axis off
subplot(3,1,[2,3])
surf(az_loc,el_loc,r_sm)
hold on
plot3(az,el,r,'.');
shading interp
axis tight


[sm_x,sm_y,sm_z] = sph2cart(az_loc,el_loc,r_sm);
sm_xy=[reshape(sm_x,numel(sm_x),1),reshape(sm_y,numel(sm_y),1)];
sm_z_fit=reshape(f(FIT,sm_xy),size(sm_z));

figure;
Delta=sm_z-sm_z_fit;
Delta(abs(Delta)>.2)=0;
surf(sm_x,sm_y,Delta);
axis tight
shading interp


end

