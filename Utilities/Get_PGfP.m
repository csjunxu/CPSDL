function [seed, PatchGroup, PGmean] = Get_PGfP(Patches,row,col,par)
off     =  (col-1)*par.maxr + row;
rmin    =   max( row-par.Win, 1 );
rmax    =   min( row+par.Win, par.maxr );
cmin    =   max( col-par.Win, 1 );
cmax    =   min( col+par.Win, par.maxc );
idx     =   par.Index(rmin:rmax, cmin:cmax);
idx     =   idx(:);
neighbor       =   Patches(:,idx);
seed       =   Patches(:,off);
dis = sum(bsxfun(@minus,neighbor, seed).^2,1);
[~,ind]   =  sort(dis);
indc        =  idx( ind( 1 : par.nlsp ) );
PatchGroup = Patches(:,indc); % or X_nl = neighbor(:,ind( 1 : par.nlsp ));
% Removes mean from patch group
PGmean = mean(PatchGroup,2);
PGmean = repmat(PGmean, [1 par.nlsp]);