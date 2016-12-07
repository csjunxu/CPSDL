%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupled Projection and Shared Dictionary Learning
% CopyRight @ Jun Xu 11/27/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
addpath('Data');
addpath('Utilities');

task = 'BID';

if strcmp(task,'BID') == 1
    load Data/GMM_RGB_P_BID_6x6_65_20161204T082637.mat;
elseif strcmp(task,'SR') == 1
    load Data/EMGM_SR_8x8_100_20160920T201654.mat;
end
%% Parameters Setting
% tunable parameters
par.lambda1         =       [0.001 0.01];
par.lambda2         =       0.01;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.rho = 0.05;
% fixed parameters
par.epsilon         =        5e-3;
par.cls_num            =    cls_num;
par.step               =    2;
par.ps                =    6;
par.ch = 3;
par.nIter           =       100;
par.t0              =       5;
par.K               =       256;
par.L               =       par.ps^2;
% Multi-Layer DL
Layer = 2;
PSNR = zeros(par.cls_num,Layer+1);
SSIM = zeros(par.cls_num,Layer+1);
for i = 1 : par.cls_num
    XC = double(Xc{i});
    XN = double(Xn{i});
    XC = XC - repmat(mean(XC), [par.ps^2*par.ch 1]);
    XN = XN - repmat(mean(XN), [par.ps^2*par.ch 1]);
    fprintf('Coupled Projection and Shared Dictioanry Learning (%s): Cluster: %d\n', task, i);
    PSNR(i ,1) = csnr( XN*255, XC*255, 0, 0 );
    SSIM(i ,1) = cal_ssim( XN*255, XC*255, 0, 0 );
    fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(i ,1), SSIM(i ,1) );
    % Training
    for L = 1:Layer
        % parameters
        param.K = par.K;
        param.iter=300;
        param.lambda = par.lambda1(L);
        param.L = par.ps^2;
        param.lambda2      =       par.lambda2;
        param.mode          = 	    2;       % penalized formulation
        param.approx=0;
        flag_initial_done = 0;
        % Initiatilization
        Pc = eye(size(XC, 1));
        Pn = eye(size(XN, 1));
        D = mexTrainDL([XN XC], param);
        Alphac = mexLasso(Pc * XC, D, param);
        Alphan = mexLasso(Pn * XN, D, param);
        [D, Pc, Pn, Alphac, Alphan] = CPSDL(Alphac, Alphan, XC, XN, D, Pc, Pn, par, param);
        Dict.D{i,L} = D;
        Dict.PC{i,L} = Pc;
        Dict.PN{i,L} = Pn;
        XN = D * Alphan;
        PSNR(i ,L+1) = csnr( XN*255, XC*255, 0, 0 );
        SSIM(i ,L+1) = cal_ssim( XN*255, XC*255, 0, 0 );
        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(i ,L+1), SSIM(i ,L+1) );
        Dict_BID_backup = sprintf('Data/CPSDL_%s_RGB_P_ML_DL_%s.mat',task,datestr(now, 30));
        save(Dict_BID_backup,'Dict');
    end
end
paramsname = sprintf('Data/params_20161207_1.mat');
save(paramsname,'par','param');