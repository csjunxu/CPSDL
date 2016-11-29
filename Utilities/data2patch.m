function patch = data2patch(data, par)
[h, w, ch]   =   size(data);
N       =  h-par.ps+1;
M       =  w-par.ps+1;
r     =  [1:par.step:N];
r     =  [r r(end)+1:N];
c     =  [1:par.step:M];
c     =  [c c(end)+1:M];
patch      =  zeros(par.ps^2*ch,length(r)*length(c), 'double');
k          =   0;
for l = 1:ch
    for i  = 1:par.ps
        for j  = 1:par.ps
            k        =  k+1;
            blk  =  data(r-1+i,c-1+j,l);
            patch(k,:)  =  blk(:)';
        end
    end
end