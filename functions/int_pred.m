function [delta,deltaA] = int_pred(x1,x2,y,p)
%INT_PRED Compute interaction predictability and additive interaction predictability.
%
%   [delta,deltaA] = int_pred(x1,x2,y,p)
%
%   Inputs:
%       x1 : first source/driver variable, N x 1 vector
%       x2 : second source/driver variable, N x 1 vector
%       y  : target variable, N x 1 vector
%       p  : maximum polynomial order
%
%   Outputs:
%       delta  : interaction predictability computed using the full model
%                including additive and interaction terms
%
%       deltaA : additive interaction predictability computed using only
%                additive source-specific terms
%
%   The function implements the quantities:
%
%       Delta  = R_1^* + R_2^* - R^*   - var(Y)
%       DeltaA = R_1^* + R_2^* - R_A^* - var(Y)
%
%   where:
%       R^*   is the residual variance of the full model
%       R_A^* is the residual variance of the additive model
%       R_1^* is the residual variance using source x1 only
%       R_2^* is the residual variance using source x2 only
%
%   The full model contains:
%       x1, x1^2, ..., x1^p
%       x2, x2^2, ..., x2^p
%       interaction terms x1^k x2^j with k+j <= p
%
%   The additive model contains only:
%       x1, x1^2, ..., x1^p
%       x2, x2^2, ..., x2^p
%
%   No interaction terms are included in the additive model.

% Ensure column vectors
x1 = x1(:);
x2 = x2(:);
y  = y(:);

% Number of samples
N = length(x1);

% -------------------------------------------------------------------------
% Build additive polynomial regressors
% -------------------------------------------------------------------------
% X will contain:
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

% -------------------------------------------------------------------------
% Build non-additive interaction regressors
% -------------------------------------------------------------------------
% XX contains all polynomial interaction terms:
%
%       x1^k * x2^j
%
% with total degree:
%
%       k + j <= p
%
% For example:
%   p = 2 -> x1*x2
%   p = 3 -> x1*x2, x1^2*x2, x1*x2^2
%   p = 4 -> x1*x2, x1^2*x2, x1*x2^2, x1^3*x2,
%            x1^2*x2^2, x1*x2^3

XX = [];

icont = 0;

for k = 1:p-1
    for j = 1:p-k
        icont = icont + 1;
        XX(:,icont) = (x1.^k) .* (x2.^j);
    end
end

% -------------------------------------------------------------------------
% Full model: joint prediction with additive and interaction terms
% -------------------------------------------------------------------------
% This approximates the unrestricted/general predictor.
%
% Model:
%   y = additive terms in x1 and x2
%       + interaction terms x1^k x2^j
%       + residual
%
% Residual variance from this model gives R^*.

[~,~,R_full] = regress(y,[X XX]);

% -------------------------------------------------------------------------
% Single-source model using x1 only
% -------------------------------------------------------------------------
% Columns used:
%   1:p       -> x1 polynomial terms
%   2p+1      -> intercept
%
% Residual variance from this model gives R_1^*.

[~,~,R_x1] = regress(y,X(:,[1:p 2*p+1]));

% -------------------------------------------------------------------------
% Single-source model using x2 only
% -------------------------------------------------------------------------
% Columns used:
%   p+1:2p    -> x2 polynomial terms
%   2p+1      -> intercept
%
% Residual variance from this model gives R_2^*.

[~,~,R_x2] = regress(y,X(:,[p+1:2*p 2*p+1]));

% -------------------------------------------------------------------------
% Additive joint model
% -------------------------------------------------------------------------
% This model includes both sources, but only through separate polynomial
% contributions. Interaction terms are excluded.
%
% Model:
%   y = h1(x1) + h2(x2) + residual
%
% Residual variance from this model gives R_A^*.

[~,~,R_add] = regress(y,X);

% -------------------------------------------------------------------------
% Compute interaction predictability measures
% -------------------------------------------------------------------------
% Delta compares the prediction error of the full joint model against
% the sum of the two single-source models.
%
% DeltaA does the same comparison, but replacing the full model with
% the additive joint model.
%
delta  = var(R_x2) + var(R_x1) - var(R_full) - var(y);
deltaA = var(R_x2) + var(R_x1) - var(R_add)  - var(y);

end