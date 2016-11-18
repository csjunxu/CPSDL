function X = rnd_smp_patches(Trainingdir, im_dir, im_num, num_patch, par)

% noisy patches per image
nper_img = zeros(1, im_num);
for ii = 1:im_num
    im = im2double(imread(fullfile(Trainingdir, im_dir(ii).name)));
    [h,w,ch] = size(im);
    if h > 512
        randh = randi(h-512);
        im = im(randh+1:randh+512,:,:);
    end
    if w > 512
        randw = randi(w-512);
        im = im(:,randw+1:randw+512,:);
    end
    IM{ii} = im;
    nper_img(ii) = numel(im);
end
nper_img = floor(nper_img*num_patch/sum(nper_img));
% extract noisy PGs
X = [];
for ii = 1:im_num
    patch_num = nper_img(ii);
    [h,w,ch] = size(IM{ii});
    par.h = h;
    par.w = w;
    par.ch = ch;
    P = sample_RGB_Patches(IM{ii}, patch_num, par);
    X = [X, P];
end