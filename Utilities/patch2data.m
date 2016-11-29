function im_pout= patch2data(Y, h,w,ch, b, s)
im_pout   =  zeros(h,w,ch, 'double');
im_wei   =  zeros(h,w,ch, 'double');
k          =   0;
N       =  h-b+1;
M       =  w-b+1;
r     =  [1:s:N];
r     =  [r r(end)+1:N];
c     =  [1:s:M];
c     =  [c c(end)+1:M];
N       =  length(r);
M       =  length(c);
for l = 1:ch
    for i  = 1:b
        for j  = 1:b
            k    =  k+1;
            im_pout(r-1+i,c-1+j,l)  =  im_pout(r-1+i,c-1+j,l) + reshape( Y(k,:)', [N M]);
            im_wei(r-1+i,c-1+j,l)  =  im_wei(r-1+i,c-1+j,l) + 1;
        end
    end
end
im_pout  =  im_pout./(im_wei+eps);