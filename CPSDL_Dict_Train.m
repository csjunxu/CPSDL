%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupled Projection and Shared Dictionary Learning
% CopyRight @ Jun Xu 09/17/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
addpath('../DSCDL_BID/Data');
addpath('Utilities');
addpath('../DSCDL_BID/SPAMS');
% addpath('SPAMS/release/mkl64');

load Data/params.mat;
load Data/EMGM_8x8_100_knnNI2BS500Train_20160722T082406.mat;
% Parameters Setting
par.rho = 0.05;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.epsilon         =        5e-3;
par.cls_num            =    cls_num;
par.step               =    2;
par.win                =    8;
par.nIter           =       100;
par.t0              =       5;
par.K               =       256;
par.L               =       par.win * par.win;
param.K = par.K;
param.iter=300;
param.lambda = par.lambda1;
param.L = par.win * par.win;
param.lambda2       =       par.lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
flag_initial_done = 0;
paramsname = sprintf('Data/params.mat');
save(paramsname,'par','param');

load Data/EMGM_8x8_100_20160917T233957.mat;
for i = 1 : par.cls_num
    XC = double(Xc{i});
    XN = double(Xn{i});
    XC = XC - repmat(mean(XC), [par.win^2 1]); 
    XN = XN - repmat(mean(XN), [par.win^2 1]);
    fprintf('Coupled Projection and Shared Dictioanry Learning : Cluster: %d\n', i);
    % Initiatilization
    Pc = eye(size(XC, 1));
    Pn = eye(size(XN, 1));
    D = mexTrainDL([XN XC], param);
    Alphac = mexLasso(Pc * XC, D, param);
    Alphan = mexLasso(Pn * XN, D, param);
    % Training
    [D, Pc, Pn] = CPSDL(Alphac, Alphan, XC, XN, D, Pc, Pn, par, param);
    Dict.D{i} = D;
    Dict.PC{i} = Pc;
    Dict.PN{i} = Pn;
    Dict_BID_backup = sprintf('Data/CPSDL_BID_Dict_backup_%s.mat',datestr(now, 30));
    save(Dict_BID_backup,'Dict');
end