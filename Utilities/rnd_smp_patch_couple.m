function [XN, XC] = rnd_smp_patch_couple(TrainingNoisy, TrainingClean, patch_size, num_patch, R_thresh)

Nim_path = fullfile(TrainingNoisy,'*.png');
Cim_path = fullfile(TrainingClean,'*.png');

Nim_dir = dir(Nim_path);
Cim_dir = dir(Cim_path);

Nim_num = length(Nim_dir);
Cim_num = length(Cim_dir);
nper_img = zeros(1, Nim_num);
for ii = 1:length(Nim_dir)
    Nim = im2double(imread(fullfile(TrainingNoisy, Nim_dir(ii).name)));
    [h,w] = size(Nim);
    if h >= 1000
        randh = randi(h-1000);
        Nim = Nim(randh+1:randh+1000,:,:);
    end
    if w >= 1000
        randw = randi(w-1000);
        Nim = Nim(:,randw+1:randw+1000,:);
    end
    nper_img(ii) = numel(Nim);
end

nper_img = floor(nper_img*num_patch/sum(nper_img));


XN = [];
XC = [];
for ii = 1:Nim_num,
    warning off;
    patch_num = nper_img(ii);
    Nim = im2double(imread(fullfile(TrainingNoisy, Nim_dir(ii).name)));
    Cim = im2double(imread(fullfile(TrainingClean, Cim_dir(ii).name)));
    [h,w] = size(Nim);
    if h >= 1000
        randh = randi(h-1000);
        Nim = Nim(randh+1:randh+1000,:,:);
        Cim = Cim(randh+1:randh+1000,:,:);
    end
    if w >= 1000
        randw = randi(w-1000);
        Nim = Nim(:,randw+1:randw+1000,:);
        Cim = Cim(:,randw+1:randw+1000,:);
    end
    imwrite(Nim,['./Data/cropN_' Nim_dir(ii).name]);
    imwrite(Cim,['./Data/cropC_' Cim_dir(ii).name]);
    [N, C] = sample_patches(Nim, Cim, patch_size, patch_num, R_thresh);
    XN = [XN, N];
    XC = [XC, C]; 
end
num_patch = size(XN,2);
patch_path = ['Training/rnd_patches_' num2str(patch_size) 'x' num2str(patch_size) '_' num2str(num_patch) '_' num2str(R_thresh) '.mat'];
save(patch_path, 'XN', 'XC');