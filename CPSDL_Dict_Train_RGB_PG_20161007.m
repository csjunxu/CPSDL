%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupled Projection and Shared Dictionary Learning
% CopyRight @ Jun Xu 09/17/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
addpath('Data');
addpath('Utilities');

task = 'BID';

load Data/params_gray_PG.mat;
% if strcmp(task,'BID') == 1
%     load ../DSCDL_BID/Data/GMM_PG_3_10_8x8_64_20161003T060541.mat;
% elseif strcmp(task,'SR') == 1
%     load Data/EMGM_SR_8x8_100_20160920T201654.mat;
% end
% Parameters Setting
par.cls_num            =    cls_num;
par.step               =    3;
par.ps                =   6;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       256;
% par.L               =       par.ps^2;
param.K = par.K;
param.lambda = par.lambda1;
param.lambda2 = par.lambda2;
param.iter=300;
% param.L = par.ps^2;
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    fprintf('Coupled Projection and Shared Dictioanry Learning (%s)::_RGB_PGs: Cluster: %d\n', task,i);
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
    Dict_BID = sprintf('Data/CPSDL_RGB_PG_6x6x3_10_33_%s_20161007.mat',task);
    save(Dict_BID,'Dict');
end