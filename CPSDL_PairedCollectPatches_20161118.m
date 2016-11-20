clear;
addpath('Data');
addpath('Utilities');

% parameters
patch_size = 6;
Patch_Channel = 3;
num_patch_N = 500000;
R_thresh = 0.05;
cls_num = 64;

% Parameters Setting
par.step               =    2;
par.Patch_Channel = Patch_Channel;
par.ps                =    patch_size;
par.S         =  2 * par.ps - 1;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K = 256;
par.L = par.ps^2;
par.cls_num = cls_num;
param.K = par.K;
param.lambda = par.lambda1;
param.iter=300;
param.L = par.ps^2;
save Data/params.mat par param;

% blind image denoising or super-resolution
task = 'BID';
% BID : blind image denoising
% SR   : super-resolution

if strcmp(task, 'BID') == 1
    num_patch_C = 5*num_patch_N;
    TrainingNoisy = '../../Projects/CVPR2016_crosschannel/ccnoise_denoised_part/';
    TrainingClean = '../../Projects/4000images/';
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch_N, num_patch_C, R_thresh, par);
elseif strcmp(task, 'BSR') == 1
    TrainingNoisy = 'TrainingImages/SRTrain91/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = 'TrainingImages/SRTrain91/'; % kNN
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch1, num_patch2, R_thresh);
elseif strcmp(task, 'SR') == 1
    upscale = 2;
    TrainingHR = 'TrainingImages/SRTrain91/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingLR = 'TrainingImages/SRTrain91/'; % kNN
    [X1, X2] = rnd_smp_patch_coupled(TrainingLR, TrainingHR, patch_size, num_patch1, R_thresh, upscale);
else
    disp('Default Task is Blind Image Denoising !');
    TrainingNoisy = '../TrainingData/ycbcrNoisy/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = '../TrainingData/BSDS500Train/'; % kNN
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch1, num_patch2, R_thresh);
end


[model, cls_idx]  =  emgm(X2, cls_num);
for c = 1 : cls_num
    idx = find(cls_idx == c);
    %     if (length(idx) >  100000)
    %          select_idx = randperm(length(idx));
    %         idx = idx(select_idx(1:100000));
    %     end
    Xn{c} = X1(:, idx);
    Xc{c} = X2(:, idx);
end
clear X1 X2;
GMM_model = ['Data/GMM_PairedPatches_' task '_' num2str(patch_size) 'x' num2str(patch_size) '_' num2str(cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','cls_num','task');