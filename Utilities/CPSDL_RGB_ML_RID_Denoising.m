function [im_out, par] = CPSDL_RGB_ML_RID_Denoising(Im_in,IM_GT,model,Dict,par,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_{alpha1,alpha2} ||P1*y1 - D*alpha1||_{2}^{2} + ||P2*y2 - D*alpha2||_{2}^{2}
% + ||P1*y1 - P2*y2||_{2}^{2} + ||alpha1||_{1} + ||alpha2||_{1}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h,w,ch] = size(Im_in);
% Initial the output image as the input IMin
im_out = Im_in;
for t = 1 : par.nInnerLoop
    if mod(t -1,2) == 0
        YH = data2patch(im_out, par);
        Num_Patches = size(YH,2);
        AN = zeros(par.K, Num_Patches);
        AC = zeros(par.K, Num_Patches);
        %% GMM: full posterior calculation
        PYZ = zeros(model.nmodels,size(YH,2));
        for i = 1:model.nmodels
            sigma = model.covs(:,:,i);
            [R,~] = chol(sigma);
            Q = R'\YH;
            PYZ(i,:)  = - sum(log(diag(R))) - dot(Q,Q,1)/2;
        end
        %% find the most likely component for each patch group
        [~,cls_idx] = max(PYZ);
        par.cls_idx = cls_idx;
        par.Num_Patches = Num_Patches;
    else
        AN = par.AN;
        AC = par.AC;
    end
    XC = data2patch(im_out,  par);
    XN = data2patch(Im_in,  par);
    meanX = repmat(mean(XC), [par.ps^2*ch 1]);
    XC = XC - meanX;
    XN = XN - meanX;
    for i = 1 : par.cls_num
        if t == 1
            idx_cluster   = find(par.cls_idx == i);
            par.idx_cluster{i} = idx_cluster;
        else
            idx_cluster = par.idx_cluster{i};
        end
        Xn    = double(XN(:, idx_cluster));
        D    = Dict.D{i};
        Pc    = Dict.PC{i};
        Pn    = Dict.PN{i};
        switch param.Case
            case 0
                Xc = Xn;
                Alphan = AN(:, idx_cluster);
                Alphac = AC(:, idx_cluster);
            case 1
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = D * Alphan;
            case 2
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = D * Alphan;
            case 3
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = Pc \ D * Alphan;
            case 4
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc_temp = Pc \ D * Alphan;
                % min_{alpha2} ||P2*y2-D*alpha2||_{2}^{2}+||alpha2||_{1}
                Alphac = mexLasso(Pc*Xc_temp, D, param);
                %% Reconstruction
                Xc = Pc \ D * Alphac;
            case 5
                if (par.nOuterLoop == 1)
                    % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                    Alphan = mexLasso(Pn*Xn, D, param);
                    % min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                    Xc_temp = Pc \ D * Alphan;
                else
                    Alphac = AC(:, idx_cluster);
                    Xc_temp = D * Alphac;
                end
                % min_{alpha2} ||P2*y2-D*alpha2||_{2}^{2}+||alpha2||_{1}
                Alphac = mexLasso(Pc*Xc_temp, D, param);
                Xn_temp = Pn \ D * Alphac;
                Alphan = mexLasso(Xn_temp, D,param);
                % Reconstruction
                Xc = Pc \ D * Alphan;
            case 6
                if (par.nOuterLoop == 1)
                    % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                    Alphan = mexLasso(Pn*Xn, D, param);
                    % min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                    Xc_temp = Pc \ D * Alphan;
                else
                    Alphac = AC(:, idx_cluster);
                    Xc_temp = D * Alphac;
                end
                % min_{alpha2} ||P2*y2-D*alpha2||_{2}^{2}+||alpha2||_{1}
                Alphac = mexLasso(Pc*Xc_temp, D, param);
                Xn_temp = Pn \ D * Alphac;
                Alphan = mexLasso(Xn_temp, D,param);
                Xc_temp2 = Pc \ D * Alphan;
                % min_{alpha2} ||P2*y2-D*alpha2||_{2}^{2}+||alpha2||_{1}
                Alphac = mexLasso(Pc*Xc_temp2, D, param);
                % Reconstruction
                Xc = Pc \ D * Alphac;
            otherwise
                disp('Not find this case!');
                break;
        end
        XC(:, idx_cluster) = Xc;
        AN(:, idx_cluster) = Alphan;
        AC(:, idx_cluster) = Alphac;
    end
    par.AN =  AN;
    par.AC = AC;
    im_out = patch2data(XC+meanX, h, w, ch, par.ps, par.step);
    par.PSNR(par.IMindex, t) = csnr( im_out*255, IM_GT*255, 0, 0 );
    par.SSIM(par.IMindex, t) = cal_ssim( im_out*255, IM_GT*255, 0, 0 );
    fprintf('nInnerLoop %d: PSNR = %2.4f, SSIM = %2.4f. \n', t, par.PSNR(par.IMindex, t), par.SSIM(par.IMindex, t) );
end