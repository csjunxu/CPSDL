function [P1, P2] = sample_patches_coupled(im1, im2, patch_size, patch_num, R_thresh, upscale)
if size(im2, 3) == 3,
    im2 = rgb2ycbcr(im2);
    im2 = im2(:, :, 1);
else
    disp('grayscale image sampled!');
end
% generate low resolution counter parts
im1 = imresize(im2, 1/upscale, 'bicubic');
im1 = imresize(im1, size(im2), 'bicubic');
[nrow, ncol] = size(im2);

x = randperm(nrow-2*patch_size-1) + patch_size;
y = randperm(ncol-2*patch_size-1) + patch_size;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

im1 = double(im1);

P1 = [];
P2 = [];
idx=1;ii=1;n=length(xrow);
while (idx < patch_num) && (ii<=n),
    row = xrow(ii);
    col = ycol(ii);
    
    patch1 = im1(row:row+patch_size-1,col:col+patch_size-1);
    patch2 = im2(row:row+patch_size-1,col:col+patch_size-1);
    
    % check if it is a stochastic patch
    p1 = patch1(:) - mean(patch1(:));
    p2 =  patch2(:) - mean(patch2(:));
    p2norm=sqrt(sum(p1.^2));
    p2_normalised=reshape(p2/p2norm,patch_size,patch_size);
    % eliminate that small variance patch
    if var(p2)>0.001
        % eliminate stochastic patch
        if dominant_measure(p2_normalised)>R_thresh
            %if dominant_measure_G(Lpatch1,Lpatch2)>R_thresh
            P1 = [P1 p1];
            P2 = [P2 p2];
            idx=idx+1;
        end
    end
    
    ii=ii+1;
end

fprintf('sampled %d patches.\r\n',patch_num);
end

function R = dominant_measure(p)
% calculate the dominant measure
% ref paper: Eigenvalues and condition numbers of random matries, 1988
% p = size n x n patch

hf1 = [-1,0,1];
vf1 = [-1,0,1]';
Gx = conv2(p, hf1,'same');
Gy = conv2(p, vf1,'same');

G=[Gx(:),Gy(:)];
[U, S, V]=svd(G);

R=(S(1,1)-S(2,2))/(S(1,1)+S(2,2));

end
