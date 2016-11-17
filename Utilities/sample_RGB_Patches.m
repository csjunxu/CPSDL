function P = sample_RGB_Patches(im, patch_num, par)

if size(im, 3) == 3,
    disp('RGB image sampled!');
else
    disp('Grayscale image sampled!');
end
[h, w, ch] = size(im);
par.h = h;
par.w = w;
par.ch = ch;
par.maxr         =  h - par.ps + 1;
par.maxc         =  w - par.ps + 1;
r         =  1:par.maxr;
par.r         =  [r r(end)+1:par.maxr];
c         =  1:par.maxc;
par.c         =  [c c(end)+1:par.maxc];

x = randperm(h-2*par.ps-1) + par.ps;
y = randperm(w-2*par.ps-1) + par.ps;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

im = double(im);

P = [];
idx=1;ii=1;n=length(xrow);
while (idx < patch_num) && (ii<=n),
    row = xrow(ii);
    col = ycol(ii);
    Patch = im(row:row+par.ps-1,col:col+par.ps-1,:);
    % check if it is a stochastic patch
    np = Patch(:) - mean(Patch(:));
%     npnorm=sqrt(sum(np.^2));
%     np_normalised=reshape(np/npnorm,[par.ps par.ps ch]);
%     % eliminate stochastic patch
%     if dominant_measure(np_normalised)>par.R_thresh
        P = [P np];
        idx=idx+1;
%     end
    
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
Gx = convn(p, hf1,'same');
Gy = convn(p, vf1,'same');

G=[Gx(:),Gy(:)];
[U, S, V]=svd(G);

R=(S(1,1)-S(2,2))/(S(1,1)+S(2,2));

end
