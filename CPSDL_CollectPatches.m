clear;
addpath('Data');
addpath('Utilities');

% parameters
patch_size = 8;
 num_patch1 = 200000;
R_thresh = 0.05;
cls_num = 100;
% blind image denoising or super-resolution
task = 'SR';
% BID : blind image denoising
% SR   : super-resolution

if strcmp(task, 'BID') == 1
    num_patch2 = 5*num_patch1;
    TrainingNoisy = 'TrainingImages/ycbcrNoisy/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = 'TrainingImages/BSDS500Train/'; % kNN
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch1, num_patch2, R_thresh);
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

run('CPSDL_Param_Setting.m');
% load TrainingImages/rnd_patches_8x8_866242_0.05_20160715T232001.mat

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
GMM_model = ['Data/EMGM_' task '_' num2str(patch_size) 'x' num2str(patch_size) '_' num2str(cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','cls_num','task');