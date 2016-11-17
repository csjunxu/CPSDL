clear;
addpath('Data'); 
addpath('Utilities');
TrainingClean = '../../Projects/4000images/';
im_path = fullfile(TrainingClean,'*.bmp');
im_dir = dir(im_path);
im_num = length(im_dir);

num_patch_C = 2000000;

%% set parameters
run('CPSDL_Param_Setting.m');
load Data/params.mat;

%% collect clean patches
XC = rnd_smp_patches(TrainingClean , im_dir, im_num, num_patch_C, par);
num_patch = size(XC,2);
patch_path = ['Data/CPSDL_RGB_CP_' num2str(par.ps) 'x' num2str(par.ps) '_' num2str(num_patch)  '_' datestr(now, 30) '.mat'];
save(patch_path, 'XC');

%%
[model, cls_idx]  =  emgm(XC, par.cls_num); 
for c = 1 : par.cls_num
    idx = find(cls_idx == c);
    if (length(idx) >  100000)
         select_idx = randperm(length(idx));
        idx = idx(select_idx(1:100000));
    end
    Xc{c} = XC(:, idx);
end
clear XC;
GMM_model = ['Data/GMM_' num2str(par.ps) 'x' num2str(par.ps) 'x' num2str(par.ch) '_' num2str(par.cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model','Xc');  