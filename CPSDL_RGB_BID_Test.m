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

load Data/CPSDL_PairedPatches_6x6x3_64.mat;
params = 'Data/params.mat';
load(params,'par','param');
Dict_SR_backup = 'Data/CPSDL_BID_Dict_backup_20161124T064021.mat';
load(Dict_SR_backup,'Dict');
for lambda2 = 0.0005
    param.lambda2 = lambda2;
    for lambda = 0.02
        param.lambda = lambda;
        for solver = 1:1:6
            param.Case = solver;
            PSNR = [];
            SSIM = [];
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
                fprintf('%s: \n',TT_im_dir(i).name);
                par.nOuterLoop = 1;
                Continue = true;
                while Continue
                    fprintf('Iter: %d \n', par.nOuterLoop);
                    [IMout, par] = CPSDL_RGB_RID_Denoising(IMin,model,Dict,par,param);
                    % calculate the PSNR and SSIM
                    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n',par.nOuterLoop, csnr( IMout*255, IM_GT*255, 0, 0 ),cal_ssim( IMout*255, IM_GT*255, 0, 0 ));
                    % Noise Level Estimation
                    nSig =NoiseEstimation(IMout*255,par.ps);
                     fprintf('The noise level is %2.4f.\n',nSig);
                    if nSig < 0.001 || par.nOuterLoop >= 10
                        Continue = false;
                    else
                        par.nOuterLoop = par.nOuterLoop + 1;
                        IMin = IMout;
                    end
                      % calculate the PSNR and SSIM
                PSNR(par.nOuterLoop,i) =   csnr( IMout*255, IM_GT*255, 0, 0 );
                SSIM(par.nOuterLoop,i)      =  cal_ssim( IMout*255, IM_GT*255, 0, 0 );
                fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n',par.nOuterLoop, PSNR(par.nOuterLoop,i),SSIM(par.nOuterLoop,i));
                end
                %  imwrite(IMout, ['C:\Users\csjunxu\Desktop\ICCV2017\cc_Results\Real_' method '\' method '_'  num2str(lambda) '_'  num2str(lambda2) '_' IMname '.png']);
            end
            mPSNR = mean(PSNR,2);
            mSSIM = mean(SSIM,2);
            mCCPSNR = mean(CCPSNR);
            mCCSSIM = mean(CCSSIM);
            save(['C:\Users\csjunxu\Desktop\ICCV2017\cc_Results\Real_' method '\' method '_CCNoise_' num2str(lambda) '_'  num2str(lambda2) '_' num2str(solver) '.mat'],'PSNR','mPSNR','SSIM','mSSIM','CCPSNR','mCCPSNR','CCSSIM','mCCSSIM');
        end
    end
end