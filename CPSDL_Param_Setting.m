% Parameters Setting
par.rho = 0.05;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.R_thresh = 0.05;
% fixed parameters
par.epsilon         =        5e-3;
par.cls_num            =    100;
par.step               =    3;
par.ps               =    6;
par.ch = 3;
par.nIter           =       100;
par.t0              =       5;
par.K               =       256;
par.L               =       par.ps^2;

param.K = par.K;
param.iter=300;
param.lambda = par.lambda1;
param.L = par.L
param.lambda2       =       par.lambda2;
param.mode          = 	    2;       % Elastic-Net
param.approx=0;
flag_initial_done = 0;
paramsname = sprintf('Data/params.mat');
save(paramsname,'par','param');