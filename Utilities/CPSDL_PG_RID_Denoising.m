function [im_out, par] = CPSDL_PG_RID_Denoising(IMin,model,Dict,par,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% min_{alpha1,alpha2} ||P1*y1 - D*alpha1||_{2}^{2} + ||P2*y2 - D*alpha2||_{2}^{2}
% + ||P1*y1 - P2*y2||_{2}^{2} + ||alpha1||_{1} + ||alpha2||_{1}
%
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h,w] = size(IMin);
% Initial the output image as the input IMin
im_out = IMin;
for t = 1 : par.nInnerLoop
    if t == 1
        psf = fspecial('gaussian', par.win+2, 2.2);
        [nDCnlYH,~,~,par] = Image2PGs( conv2(im_out, psf, 'same') - im_out, par);
        AN = zeros(par.K, size(nDCnlYH, 2));
        AC = zeros(par.K, size(nDCnlYH, 2));
        %% GMM: full posterior calculation
        nPG = size(nDCnlYH,2)/par.nlsp; % number of PGs
        PYZ = zeros(model.nmodels,nPG);
        for i = 1:model.nmodels
            sigma = model.covs(:,:,i);
            [R,~] = chol(sigma);
            Q = R'\nDCnlYH;
            TempPYZ = - sum(log(diag(R))) - dot(Q,Q,1)/2;
            TempPYZ = reshape(TempPYZ,[par.nlsp nPG]);
            PYZ(i,:) = sum(TempPYZ);
        end
        %% find the most likely component for each patch group
        [~,cls_idx] = max(PYZ);
        cls_idx=repmat(cls_idx, [par.nlsp 1]);
        cls_idx = cls_idx(:);
        [idx,  s_idx] = sort(cls_idx);
        idx2 = idx(1:end-1) - idx(2:end);
        seq = find(idx2);
        seg = [0; seq; length(cls_idx)];
    else
        AN = par.AN;
        AC = par.AC;
    end
    %%  Image to PGs
    [nDCnlXC,blk_arrXC,DCXC,par] = Image2PGs( im_out, par);
    [nDCnlXN,~,~,par] = Image2PGs( IMin_y, par);
    X_hat = zeros(par.ps^2,par.maxr*par.maxc,'double');
    W = zeros(par.ps^2,par.maxr*par.maxc,'double');
    for   i  = 1:length(seg)-1
        if par.nOuterLoop == 1
            idx_cluster   = s_idx(seg(i)+1:seg(i+1));
            par.idx_cluster{i} = idx_cluster;
        else
            idx_cluster = par.idx_cluster{i};
        end
        cls       =   cls_idx(idx_cluster(1));
        Xc    = nDCnlXC(:, idx_cluster);
        Xn    = nDCnlXN(:, idx_cluster);
        D    = Dict.D{cls};
        Pc    = Dict.PC{cls};
        Pn    = Dict.PN{cls};
        switch param.Case
            case -1
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = D * Alphan;
            case 0
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = D * Alphan;
            case 1
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % no use
                Alphac = Alphan;
                % Reconstruction by min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc = Pc \ D * Alphan;
            case 2
                % min_{alpha1} ||P1*y1-D*alpha1||_{2}^{2}+||alpha1||_{1}
                Alphan = mexLasso(Pn*Xn, D, param);
                % min_{y2} ||P1*D*alpha1-P2*y2||_{2}^{2}
                Xc_temp = Pc \ D * Alphan;
                % min_{alpha2} ||P2*y2-D*alpha2||_{2}^{2}+||alpha2||_{1}
                Alphac = mexLasso(Pc*Xc_temp, D, param);
                %% Reconstruction
                Xc = Pc \ D * Alphac;
            case 3
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
            case 4
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
        nDCnlXC(:, idx_cluster) = Xc;
        AN(:, idx_cluster) = Alphan;
        AC(:, idx_cluster) = Alphac;
        X_hat(:,blk_arrXC(idx)) = X_hat(:,blk_arrXC(idx)) + nDCnlXC(:, idx) + DCXC(:,idx);
        W(:,blk_arrXC(idx))     = bsxfun(@plus,W(:,blk_arrXC(idx)),ones(par.ps^2,1));
    end
    par.AN =  AN;
    par.AC = AC;
    %% PGs to Image
    im_out = PGs2Image(X_hat,W,par);
    fprintf('nInnerLoop: %d, PSNR = %2.4f, SSIM = %2.4f. \n', t, csnr( im_out*255, IM_GT_y*255, 0, 0 ),cal_ssim( im_out*255, IM_GT_y*255, 0, 0 ));
end