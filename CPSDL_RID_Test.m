clear;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');
addpath('GatingNetwork');
Original_image_dir = './TestingImages/';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

load Data/EMGM_8x8_100_20160917T233957.mat;
params = 'Data/params.mat';
load(params,'par','param');
par.cls_num = 100;
param.Case = 1;
Dict_SR_backup = 'Data/CPSDL_BID_Dict_backup_20160919T125001.mat';
load(Dict_SR_backup,'Dict');
type = 'middle';
% 'random';
% 'all';
% 'middle';

for i = 1:im_num
    IMin=im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    S = regexp(im_dir(i).name, '\.', 'split');
    IMname = S{1};
    [h,w,ch] = size(IMin);
    hh = [0:1000:h,h];
    ww = [0:1000:w,w];
    num_part = 0;
    if strcmp(type, 'all')
        listh = 1 : length(hh)-1;
        listw = 1 : length(ww)-1;
        IMout = zeros(h,w,ch);
    elseif strcmp(type, 'random')
        listh = randi([1, length(hh)-1],1,1);
        listw = randi([1, length(ww)-1],1,1);
        IMout = zeros(min(h,1000),min(w,1000),ch);
    elseif strcmp(type, 'middle')
        listh = floor(median(1:length(hh)-1));
        listw = floor(median(1:length(ww)-1));
        IMout = zeros(min(h,1000),min(w,1000),ch);
    end
    fprintf('%s: \n',im_dir(i).name);
    %%
    for nh = listh
        for nw = listw
            num_part = num_part + 1;
            IMin_part = IMin(hh(nh)+1:hh(nh+1),ww(nw)+1:ww(nw+1),:);
            % color or gray image
            if ch==1
                IMin_part_y = IMin_part;
            else
                % change color space, work on illuminance only
                IMin_part_ycbcr = rgb2ycbcr(IMin_part);
                IMin_part_y = IMin_part_ycbcr(:, :, 1);
                IMin_part_cb = IMin_part_ycbcr(:, :, 2);
                IMin_part_cr = IMin_part_ycbcr(:, :, 3);
            end
            %%
            par.nOuterLoop = 1;
            Continue = true;
            while Continue
                fprintf('Iter: %d \n', par.nOuterLoop);
                [IMout_part_y, par] = CPSDL_RID_Denoising(IMin_part_y,model,Dict,par,param);
                % Noise Level Estimation
                nSig = NoiseLevel(IMout_part_y*255);
                fprintf('The noise level is %2.4f.\n',nSig);
                if nSig < 0.001 || par.nOuterLoop >= 10
                    Continue = false;
                else
                    par.nOuterLoop = par.nOuterLoop + 1;
                    IMin_part_y = IMout_part_y;
                end
            end
            if ch==1
                IMout_part = IMout_part_y;
            else
                IMout_part_ycbcr = zeros(size(IMin_part));
                IMout_part_ycbcr(:, :, 1) = IMout_part_y;
                IMout_part_ycbcr(:, :, 2) = IMin_part_cb;
                IMout_part_ycbcr(:, :, 3) = IMin_part_cr;
                IMout_part = ycbcr2rgb(IMout_part_ycbcr);
            end
            IMout(hh(nh)+1:hh(nh+1),ww(nw)+1:ww(nw+1),:) = IMout_part;
        end
    end
    %% output
    imwrite(IMout, ['./CPSDL_RID/CPSDL_BID_' num2str(param.Case) '_' IMname '.png']);
end