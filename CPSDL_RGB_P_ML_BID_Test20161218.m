clear;
addpath('Data');
addpath('Utilities');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

method = 'CPSDL';

load Data/GMM_RGB_P_BID_6x6_65_20161204T082637.mat;
params = 'Data/params.mat';
load(params,'par','param');
Dict_SR_backup = 'Data/CPSDL_BID_RGB_P_ML_4_DL_20161218T033657.mat';
load(Dict_SR_backup,'Dict');

for lambda1 = [0.005 0.01]
    for lambda2 = [0.005 0.01]
        for lambda3 = [0.03 0.05]
            for lambda4 = [0.03 0.05]
                par.lambda = [lambda1,lambda2,lambda3,lambda4];
                par.Layer = length(par.lambda);
                for solver = [1 2 3 4 5 6]
                    param.Case = solver;
                    CCPSNR = [];
                    CCSSIM = [];
                    for i = 1:im_num
                        IMin = im2double(imread(fullfile(TT_Original_image_dir,TT_im_dir(i).name) ));
                        IM_GT = im2double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
                        CCPSNR = [CCPSNR csnr( IMin*255,IM_GT*255, 0, 0 )];
                        CCSSIM = [CCSSIM cal_ssim( IMin*255, IM_GT*255, 0, 0 )];
                        fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', CCPSNR(end), CCSSIM(end));
                        S = regexp(TT_im_dir(i).name, '\.', 'split');
                        IMname = S{1};
                        [h,w,ch] = size(IMin);
                        par.IMindex = i;
                        [IMout, par] = CPSDL_RGB_P_ML_RID_Denoising(IMin,IM_GT,model,Dict,par,param);
                    end
                    %  imwrite(IMout, ['C:\Users\csjunxu\Desktop\ICCV2017\cc_Results\Real_' method '\' method '_'  num2str(lambda) '_'  num2str(lambda2) '_' IMname '.png']);
                end
                PSNR = par.PSNR;
                SSIM = par.SSIM;
                mPSNR = mean(par.PSNR,2);
                mSSIM = mean(par.SSIM,2);
                mCCPSNR = mean(CCPSNR);
                mCCSSIM = mean(CCSSIM);
                save([method '_CCNoise_' num2str(lambda1) '_' num2str(lambda2) '_' num2str(lambda3) '_' num2str(lambda4) '_' num2str(solver) '.mat'],'PSNR','mPSNR','SSIM','mSSIM','CCPSNR','mCCPSNR','CCSSIM','mCCSSIM');
            end
        end
    end
end