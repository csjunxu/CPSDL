%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Main Function of Coupled Projection and Shared Dictionary Learning
% Input :
% Alpha1, Alpha2 : Initial sparse coefficient of two domains
% X1, X2               : Image Data Pairs of two domains
% D                      : Initial Dictionaries
% P1 , P2              : Initial Projection Matrix
% par                    : Parameters for CPSDL
% param               : Parameters for mexLasso function 
% Output :
% Alpha1, Alpha2  : Output sparse coefficient of two domains
% D                       : Learned Shared Dictionary
% P1, P2               : Output Projection Matrix for X1, X2
% CopyRight @ Jun Xu 09/17/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [D, P1, P2, Alpha2] = CPSDL(Alpha1, Alpha2, X1, X2, D, P1, P2, par, param)

f = 0;
% Iteratively solve P D A

for t = 1 : par.nIter
    %% Updating Ps and Pp
    P1 = (1 - par.rho) * P1  + par.rho * [D*Alpha1 par.mu*P2*X2] * [X1 par.mu*X1]' / ( [X1 par.mu*X1]*[X1 par.mu*X1]' + par.nu * eye(size(X1, 1)));
    P2 = (1 - par.rho) * P2  + par.rho * [D*Alpha2 par.mu*P1*X1] * [X2 par.mu*X2]' / ( [X2 par.mu*X2]*[X2 par.mu*X2]' + par.nu * eye(size(X2, 1)));
 
    %% Updating Alphas and Alphap
    Alpha1 = mexLasso(P1 * X1, D, param);
    Alpha2 = mexLasso(P2 * X2, D, param);
    dictSize = par.K;

    %% Updating D
    for i=1:dictSize
       ai        =    [Alpha2(i,:) Alpha1(i,:)];
       Y         =   [P2*X2 P1*X1] - D * [Alpha2 Alpha1] + D(:,i) * ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       D(:,i)    =    di;
    end

    %% Find if converge (NEED MODIFICATION)

    F1 = P1 * X1 - D * Alpha1;
    F1 = F1(:)'*F1(:) / 2;
    F2 = par.lambda1 *  norm(Alpha1, 1);    
    F3 = P2 * X2 - P1 * X1; 
    F3 = F3(:)'*F3(:) / 2;
    F4 = par.nu * norm(P1, 'fro');
    f1 = 1 / 2 * F1 + F2 + par.mu * (F3 + F4);
    
    F1 = P2 * X2 - D * Alpha2;
    F1 = F1(:)'*F1(:) / 2;
    F2 = par.lambda1 *  norm(Alpha2, 1);    
    F3 = P2 * X2 - P1 * X1;
    F3 = F3(:)'*F3(:) / 2;
    F4 = par.nu * norm(P2, 'fro');  %%
    f2 = 1 / 2 * F1 + F2 + par.mu * (F3 + F4);
    
    %% if converge then break
    f_prev = f;
    f = f1 + f2;
    if (abs(f_prev - f) / f < par.epsilon)
        break;
    end
    fprintf('Energy %d: %d\n', t, f);
    save tempCPSDL_BID_Dict D P2 P1 P2 P1 par param;
end
    
