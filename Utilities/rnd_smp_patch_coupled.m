function [X1, X2] = rnd_smp_patch_coupled(Training1, Training2, patch_size, num_patch, R_thresh, upscale)

im_path1 = fullfile(Training1,'*.bmp');
im_path2 = fullfile(Training2,'*.bmp');

im_dir1 = dir(im_path1);
im_dir2 = dir(im_path2);

im_num1 = length(im_dir1);
im_num2 = length(im_dir2);
nper_img = zeros(1, im_num1);
for ii = 1:length(im_dir1)
    im1 = im2double(imread(fullfile(Training1, im_dir1(ii).name)));
    [h,w,ch] = size(im1);
    if h >= 1000
        randh = randi(h-1000);
        im1 = im1(randh+1:randh+1000,:,:);
    end
    if w >= 1000
        randw = randi(w-1000);
        im1 = im1(:,randw+1:randw+1000,:);
    end
    nper_img(ii) = numel(im1);
end

nper_img = floor(nper_img*num_patch/sum(nper_img));

X1 = [];
X2 = [];
for ii = 1:im_num1,
    warning off;
    patch_num = nper_img(ii);
    im1 = im2double(imread(fullfile(Training1, im_dir1(ii).name)));
    im2 = im2double(imread(fullfile(Training2, im_dir2(ii).name)));
    [h,w,ch] = size(im1);
    if h >= 1000
        randh = randi(h-1000);
        im1 = im1(randh+1:randh+1000,:,:);
        im2 = im2(randh+1:randh+1000,:,:);
    end
    if w >= 1000
        randw = randi(w-1000);
        im1 = im1(:,randw+1:randw+1000,:);
        im2 = im2(:,randw+1:randw+1000,:);
    end
    [P1, P2] = sample_patches_coupled(im1, im2, patch_size, patch_num, R_thresh, upscale);
    X1 = [X1, P1];
    X2 = [X2, P2]; 
end
num_patch = size(X1,2);
patch_path = ['Data/rnd_SR_patches_' num2str(patch_size) 'x' num2str(patch_size) '_' num2str(num_patch) '_' num2str(R_thresh) '.mat'];
save(patch_path, 'X1', 'X2');