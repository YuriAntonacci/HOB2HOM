function ss = struct_syn(x1,x2,y,p,surr_sign,surr_p)
%STRUCT_SYN Compute structural synergy from additive and full polynomial models.
%
%   ss = struct_syn(x1,x2,y,p,surr_sign,surr_p)
%
%   Inputs:
%       x1        : first source/driver variable, N x 1 vector
%       x2        : second source/driver variable, N x 1 vector
%       y         : target variable, N x 1 vector
%       p         : maximum polynomial order
%       surr_sign : surrogate flag for testing the significance of S_s
%                   0 -> original data
%                   1 -> shuffle all interaction terms
%       surr_p    : surrogate flag for testing the contribution of order-p terms
%                   0 -> original data
%                   1 -> shuffle only the terms introduced at order p
%
%   Output:
%       ss        : structural synergy estimate
%
%   The function computes:
%
%       S_s = R_A^* - R^*
%
%   where:
%       R_A^* is the residual variance of the additive model
%       R^*   is the residual variance of the full model including
%             polynomial interaction terms.
%
%   The additive model contains:
%
%       x1, x1^2, ..., x1^p,
%       x2, x2^2, ..., x2^p,
%       intercept.
%
%   The full model contains all additive terms plus interaction terms:
%
%       x1^k x2^j, with k+j <= p.
%
%   Surrogate modes:
%
%   1) If surr_sign == 1:
%      all interaction terms are randomly permuted. This destroys the
%      correspondence between the interaction regressors and the target,
%      while leaving the additive source terms unchanged. This surrogate
%      tests the null hypothesis that interaction terms do not provide an
%      additional non-additive predictive contribution, i.e. S_s = 0.
%
%   2) If surr_p == 1:
%      only the terms newly introduced at polynomial order p are randomly
%      permuted. These include:
%
%          x1^p, x2^p,
%          and interaction terms x1^k x2^j with k+j = p.
%
%      Lower-order terms are kept unchanged. This surrogate tests whether
%      moving from order p-1 to order p provides a significant increase in
%      structural synergy.
%
%   Note:
%      If both surr_sign and surr_p are set to 1, surr_sign takes priority.

% Ensure column vectors
x1 = x1(:);
x2 = x2(:);
y  = y(:);

% Number of samples
N = length(x1);

% -------------------------------------------------------------------------
% Build additive polynomial regressors
% -------------------------------------------------------------------------
% X contains:
%   columns 1:p       -> powers of x1
%   columns p+1:2p    -> powers of x2
%   column 2p+1       -> intercept
%
% Therefore:
%   X = [x1, x1^2, ..., x1^p, x2, x2^2, ..., x2^p, 1]

X = [];

icont = 0;

% Polynomial terms of x1
for k = 1:p
    icont = icont + 1;
    X(:,icont) = x1.^k;
end

% Polynomial terms of x2
for k = 1:p
    icont = icont + 1;
    X(:,icont) = x2.^k;
end

% Intercept term
icont = icont + 1;
X(:,icont) = ones(N,1);

% If the order-p surrogate test is required, store the indices of the
% additive terms introduced at order p:
%
%   x1^p is column p
%   x2^p is column 2p
%
% The intercept is not shuffled.
if surr_p == 1
    indX_surr_p = [p 2*p];
end

% -------------------------------------------------------------------------
% Build polynomial interaction regressors
% -------------------------------------------------------------------------
% XX contains interaction terms:
%
%       x1^k * x2^j
%
% with:
%
%       k >= 1, j >= 1, and k+j <= p.
%
% Examples:
%   p = 2:
%       x1*x2
%
%   p = 3:
%       x1*x2, x1^2*x2, x1*x2^2
%
%   p = 4:
%       x1*x2, x1^2*x2, x1*x2^2,
%       x1^3*x2, x1^2*x2^2, x1*x2^3

XX = [];

icont = 0;

% If the order-p surrogate test is required, this vector will store the
% indices of the interaction terms introduced at order p, i.e. those with
% total degree k+j = p.
if surr_p == 1
    indXX_surr_p = [];
end

for k = 1:p-1
    for j = 1:p-k

        icont = icont + 1;

        % Interaction monomial
        XX(:,icont) = (x1.^k) .* (x2.^j);

        % Store only the interaction terms whose total degree is p.
        % These are the new interaction terms added when moving from
        % polynomial order p-1 to polynomial order p.
        if surr_p == 1
            if (k + j == p)
                indXX_surr_p = [indXX_surr_p icont];
            end
        end

    end
end

% -------------------------------------------------------------------------
% Apply surrogate shuffling if required
% -------------------------------------------------------------------------

if surr_sign == 1

    % Surrogate test for the significance of S_s:
    % shuffle all interaction terms, leaving additive terms unchanged.
    ind = randperm(N);
    XX = XX(ind,:);

elseif surr_p == 1

    % Surrogate test for the contribution of order-p terms:
    % shuffle only the additive and interaction terms introduced at order p.
    ind = randperm(N);

    % Shuffle x1^p and x2^p
    X(:,indX_surr_p) = X(ind,indX_surr_p);

    % Shuffle interaction terms with total degree p
    if ~isempty(indXX_surr_p)
        XX(:,indXX_surr_p) = XX(ind,indXX_surr_p);
    end

end

% -------------------------------------------------------------------------
% Additive model
% -------------------------------------------------------------------------
% This model contains only separate source-specific polynomial terms:
%
%       y = h1(x1) + h2(x2) + residual
%
% No interaction terms are included.
%
% The residual variance estimates R_A^*.

[~,~,R_add] = regress(y,X);

% -------------------------------------------------------------------------
% Full model
% -------------------------------------------------------------------------
% This model contains both additive terms and interaction terms:
%
%       y = h1(x1) + h2(x2) + interaction terms + residual
%
% The residual variance estimates R^*.

[~,~,R_full] = regress(y,[X XX]);

% -------------------------------------------------------------------------
% Structural synergy
% -------------------------------------------------------------------------
% Structural synergy is the reduction in residual variance obtained by
% allowing non-additive interaction terms:
%
%       S_s = R_A^* - R^*
%
% If S_s > 0, the full model predicts the target better than the additive
% model, indicating a non-additive predictive component.

ss = var(R_add) - var(R_full);

end