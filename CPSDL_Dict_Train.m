%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupled Projection and Shared Dictionary Learning
% CopyRight @ Jun Xu 09/17/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
addpath('Data');
addpath('Utilities');

task = 'BID';

load Data/params.mat;
if strcmp(task,'BID') == 1
    load Data/GMM_PG_8x8_100_20160930T014557.mat;
elseif strcmp(task,'SR') == 1
    load Data/EMGM_SR_8x8_100_20160920T201654.mat;
end
% %% Parameters Setting
% % tunable parameters
% par.rho = 0.05;
% par.lambda1         =       0.01;
% par.lambda2         =       0.01;
% par.mu              =       0.01;
% par.sqrtmu          =       sqrt(par.mu);
% par.nu              =       0.1;
% % fixed parameters
% par.epsilon         =        5e-3;
% par.cls_num            =    cls_num;
% par.step               =    2;
% par.win                =    8;
% par.nIter           =       100;
% par.t0              =       5;
% par.K               =       256;
% par.L               =       par.win * par.win;
% param.K = par.K;
% param.iter=300;
% param.lambda = par.lambda1;
% param.L = par.win * par.win;
% param.lambda2       =       par.lambda2;
% param.mode          = 	    2;       % penalized formulation
% param.approx=0;
% flag_initial_done = 0;
% paramsname = sprintf('Data/params.mat');
% save(paramsname,'par','param');

for i = 1 : par.cls_num
    XC = double(Xc{i});
    XN = double(Xn{i});
    XC = XC - repmat(mean(XC), [par.win^2 1]);
    XN = XN - repmat(mean(XN), [par.win^2 1]);
    fprintf('Coupled Projection and Shared Dictioanry Learning (%s): Cluster: %d\n', task, i);
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
    Dict_BID_backup = sprintf('Data/CPSDL_PG_%s_Dict_backup_%s.mat',task,datestr(now, 30));
    save(Dict_BID_backup,'Dict');
end