clear;
addpath('Data');
addpath('Utilities');

% parameters
patch_size = 8; 
Patch_Channel = 1;
num_patch_N = 200000;
num_patch_C = 5*num_patch_N;
R_thresh = 0.05;
cls_num = 64;

% Parameters Setting
par.nlsp = 10;
par.step               =    2;
par.Patch_Channel = Patch_Channel;
par.patch_size                =    patch_size;
par.S         =  2 * par.patch_size - 1;
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
par.L = par.patch_size * par.patch_size;
par.cls_num = cls_num;
param.K = par.K;
param.lambda = par.lambda1;
param.iter=300; 
param.L = par.patch_size * par.patch_size;
save Data/params_gray_PG.mat par param;

% blind image denoising or super-resolution
task = 'BID';
% BID : blind image denoising
% SR   : super-resolution

if strcmp(task, 'BID') == 1
    num_patch2 = 5*num_patch1;
    TrainingNoisy = 'TrainingImages/RGBNoisy/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = 'TrainingImages/RGBBSDS500train/'; % kNN
    [X1, X2] = rnd_smp_PG_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch1, num_patch2, R_thresh);
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

run('CPSDL_PG_Param_Setting.m');
% load TrainingImages/rnd_patches_8x8_866242_0.05_20160715T232001.mat

[model, llh, cls_idx]  =  empggm(X2, cls_num, par.nlsp);
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
GMM_model = ['Data/GMM_PG_' task '_' num2str(patch_size) 'x' num2str(patch_size) '_' num2str(cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','cls_num','task');