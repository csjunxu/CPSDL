clear;
addpath('Data');
addpath('Utilities');

% parameters
num_patch_N = 10000;
cls_num = 100;
% blind image denoising or super-resolution
task = 'BID';
% BID : blind image denoising
% SR   : super-resolution
load Data/params.mat;
%%
if strcmp(task, 'BID') == 1
    TrainingNoisy = '../../Projects/CVPR2016_crosschannel/ccnoise_denoised_part/';
    im_path = fullfile(TrainingNoisy,'*real.png');
    im_dir = dir(im_path);
    im_num = length(im_dir);
    XN = rnd_smp_patches(TrainingNoisy, im_dir, im_num, num_patch_N, par);
    num_patch = size(XN,2);
    patch_path = ['Data/CPSDL_RGB_CP_' num2str(par.ps) 'x' num2str(par.ps) '_' num2str(num_patch)  '_' datestr(now, 30) '.mat'];
    save(patch_path, 'XN');
elseif strcmp(task, 'BSR') == 1
    TrainingNoisy = 'TrainingImages/SRTrain91/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = 'TrainingImages/SRTrain91/'; % kNN
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch_N, num_patch_C, R_thresh);
elseif strcmp(task, 'SR') == 1
    upscale = 2;
    TrainingHR = 'TrainingImages/SRTrain91/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingLR = 'TrainingImages/SRTrain91/'; % kNN
    [X1, X2] = rnd_smp_patch_coupled(TrainingLR, TrainingHR, patch_size, num_patch_N, R_thresh, upscale);
else
    disp('Default Task is Blind Image Denoising !');
    TrainingNoisy = '../TrainingData/ycbcrNoisy/';
    % TrainingClean = '../TrainingData/ycbcrDenoised/'; % coupled
    TrainingClean = '../TrainingData/BSDS500Train/'; % kNN
    [X1, X2] = rnd_smp_patch_kNN(TrainingNoisy, TrainingClean, patch_size, num_patch_N, num_patch_C, R_thresh);
end


load ../../CVPR2017/CVPR2017_Guided/PG-GMM_TrainingCode/PGGMM_RGB_6x6_3_win15_nlsp10_delta0.002_cls33.mat;


% %% GMM: full posterior calculation
% nPG = size(XN,2)/par.nlsp; % number of PGs
% PYZ = zeros(model.nmodels,nPG);
% for i = 1:model.nmodels
%     sigma = model.covs(:,:,i);
%     [R,~] = chol(sigma);
%     Q = R'\XN;
%     TempPYZ = - sum(log(diag(R))) - dot(Q,Q,1)/2;
%     TempPYZ = reshape(TempPYZ,[par.nlsp nPG]);
%     PYZ(i,:) = sum(TempPYZ);
% end
% % find the most likely component for each patch group
% [~,cls_idx] = max(PYZ);
% cls_idx=repmat(cls_idx, [par.nlsp 1]);
% cls_idx = cls_idx(:);
% [idx,  s_idx] = sort(cls_idx);
% idx2 = idx(1:end-1) - idx(2:end);
% seq = find(idx2);
% seg = [0; seq; length(cls_idx)];

%% GMM: full posterior calculation
nPG = size(XN,2); 
PYZ = zeros(model.nmodels,nPG);
for i = 1:model.nmodels
    sigma = model.covs(:,:,i);
    [R,~] = chol(sigma);
    Q = R'\XN;
    PYZ(i,:) = - sum(log(diag(R))) - dot(Q,Q,1)/2;
end
% find the most likely component for each patch group
[~,cls_idx] = max(PYZ);
cls_idx = cls_idx(:);
[idx,  s_idx] = sort(cls_idx);
idx2 = idx(1:end-1) - idx(2:end);
seq = find(idx2);
seg = [0; seq; length(cls_idx)];

%%
load ../../CVPR2017/CVPR2017_Guided/PG-GMM_TrainingCode/Kodak24_PGs_6x6_3_10_33.mat;
for   i = 1:length(seg)-1
    idx    =   s_idx(seg(i)+1:seg(i+1));
    cls =   cls_idx(idx(1));
    % given noisy patches, search corresponding clean ones via k-NN
    NPG = XN(:,idx);
    CPG = Xc{cls};
    PGIDX = knnsearch(NPG', CPG');
    Xn{cls} = XN(:, idx);
    Xc{cls} = CPG(:,PGIDX);
end
%% save model
GMM_model = ['Data/GMM_RGB_P_' num2str(par.ps) 'x' num2str(par.ps) '_' num2str(cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','cls_num');