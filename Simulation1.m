%% Estimation procedure following the theoretical examples
%
% This script reproduces the two theoretical examples described in Sect. 3:
%
%   1) Linear Gaussian system:
%        Y = a X1 + b X2 + U
%
%      In this case, interaction predictability Delta can be positive or
%      negative depending on the source correlation r12, but the optimal
%      predictor is additive. Therefore:
%
%        Delta_A = Delta
%        S_s     = 0
%
%      Any non-zero estimated S_s is therefore due to finite-sample effects
%      and/or estimation variability.
%
%   2) Non-linear multiplicative system:
%        Y = c X1 X2 + U
%
%      In this case, the target contains a genuine non-additive source
%      interaction. Therefore structural synergy is theoretically positive
%      for |r12| < 1:
%
%        S_s > 0
%
% The script compares theoretical and estimated values of Delta, Delta_A,
% and S_s, and performs a surrogate significance test for S_s.

clear all; close all; clc;

%% General simulation parameters

N = 1000;          % number of samples
r12 = -0.2;        % prescribed source-source correlation
a = 0.8;           % linear coefficient for X1
b = a;             % linear coefficient for X2
c = a;             % multiplicative coefficient for the non-linear model
p = 2;             % maximum polynomial order used in the regression models

Nsurr = 100;       % number of surrogate realizations
alpha = 0.05;      % significance level

rng(1);            % reproducibility

%% Generate correlated Gaussian sources X1 and X2
%
% X1 and X2 are sampled from a bivariate Gaussian distribution with
% zero mean, unit variances, and prescribed correlation r12.
%
% Note: the empirical sample correlation r12_e will be close to, but not
% exactly equal to, the prescribed value r12 because N is finite.

mu = [0 0];
Sigma = [1 r12; r12 1];

DATA = mvnrnd(mu, Sigma, N);

X1 = DATA(:,1);
X2 = DATA(:,2);

% Empirical source-source correlation
r12_e = corr(X1, X2);

% Gaussian independent noise
U = randn(N,1);

fprintf('\nEmpirical source correlation: prescribed r12 = %.4f, estimated r12 = %.4f\n', ...
    r12, r12_e);

%% =========================================================
%  Example 1: Linear Gaussian system
%  Y = a X1 + b X2 + U
% ==========================================================

Y_lin = a.*X1 + b.*X2 + U;

% Check covariance matrix consistency.
% For the linear Gaussian model, the covariance matrix of [Y X1 X2]
% should be positive definite.
SIGMA_lin = cov([Y_lin X1 X2]);
D_lin = det(SIGMA_lin);

% Theoretical values.
% Eq. (23): Delta = -r12^2(a^2+b^2) - 2ab r12
delta_th_lin  = -r12^2*(a^2+b^2) - 2*a*b*r12;
deltaA_th_lin = delta_th_lin;
Ss_th_lin     = delta_th_lin - deltaA_th_lin;

% Estimated values
[delta_lin, deltaA_lin] = int_pred(X1, X2, Y_lin, p);
Ss_est_lin = delta_lin - deltaA_lin;

% Surrogate significance test for S_s.
% Null hypothesis: interaction terms do not provide an additional
% non-additive predictive contribution, i.e. S_s = 0.
Ss_surr_lin = zeros(Nsurr,1);

for j = 1:Nsurr
    Ss_surr_lin(j) = struct_syn_v2(X1, X2, Y_lin, p, 1);
end

% Empirical upper-tail p-value with +1 correction
pval_lin = (sum(Ss_surr_lin >= Ss_est_lin) + 1) / (Nsurr + 1);

% Percentile-based threshold
thr_lin = prctile(Ss_surr_lin, 100*(1-alpha));

% Significance decision
is_significant_lin = Ss_est_lin > thr_lin;

% Print results
fprintf('\n=== Linear Gaussian system ===\n');
fprintf('Model: Y = a X1 + b X2 + U\n');
fprintf('Parameters: a = %.4f, b = %.4f, r12 = %.4f, p = %d\n', a, b, r12, p);
fprintf('det(cov[Y X1 X2]) = %.6f\n\n', D_lin);

fprintf('%-12s %15s %15s %15s\n', 'Measure', 'Theoretical', 'Estimated', 'Abs. error');
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Delta',   delta_th_lin,  delta_lin,   abs(delta_lin   - delta_th_lin));
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Delta_A', deltaA_th_lin, deltaA_lin,  abs(deltaA_lin  - deltaA_th_lin));
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Ss',      Ss_th_lin,     Ss_est_lin,  abs(Ss_est_lin - Ss_th_lin));

fprintf('\n--- Surrogate significance test for Ss ---\n');
fprintf('Nsurr                    = %d\n', Nsurr);
fprintf('alpha                    = %.4f\n', alpha);
fprintf('Surrogate threshold      = %.6f\n', thr_lin);
fprintf('Empirical p-value        = %.6f\n', pval_lin);
fprintf('Ss estimated             = %.6f\n', Ss_est_lin);

if is_significant_lin
    fprintf('Result                   = SIGNIFICANT (Ss > threshold)\n');
else
    fprintf('Result                   = NOT significant (Ss <= threshold)\n');
end

fprintf('\n');

%% =========================================================
%  Example 2: Non-linear multiplicative system
%  Y = c X1 X2 + U
% ==========================================================

Y_nonlin = c.*X1.*X2 + U;

% Check source covariance matrix consistency.
% For the non-linear multiplicative model, Y is not Gaussian; therefore,
% the relevant positive-definiteness condition concerns the source
% covariance matrix.
SIGMA_src = cov([X1 X2]);
D_src = det(SIGMA_src);

% Theoretical values.
% Eq. (24): Delta = c^2(1 - 3 r12^2)
delta_th_nonlin = c^2*(1 - 3*r12^2);

% Eq. (25): Delta_A = Delta - c^2(1-r12^2)^2/(1+r12^2)
deltaA_th_nonlin = c^2*(1 - 3*r12^2) ...
                 - c^2*((1-r12^2)^2/(1+r12^2));

% Eq. (26): S_s = Delta - Delta_A
Ss_th_nonlin = delta_th_nonlin - deltaA_th_nonlin;

% Estimated values
[delta_nonlin, deltaA_nonlin] = int_pred(X1, X2, Y_nonlin, p);
Ss_est_nonlin = delta_nonlin - deltaA_nonlin;

% Surrogate significance test for S_s
Ss_surr_nonlin = zeros(Nsurr,1);

for j = 1:Nsurr
    Ss_surr_nonlin(j) = struct_syn_v2(X1, X2, Y_nonlin, p, 1);
end

% Empirical upper-tail p-value with +1 correction
pval_nonlin = (sum(Ss_surr_nonlin >= Ss_est_nonlin) + 1) / (Nsurr + 1);

% Percentile-based threshold
thr_nonlin = prctile(Ss_surr_nonlin, 100*(1-alpha));

% Significance decision
is_significant_nonlin = Ss_est_nonlin > thr_nonlin;

% Print results
fprintf('\n=== Non-linear multiplicative system ===\n');
fprintf('Model: Y = c X1 X2 + U\n');
fprintf('Parameters: c = %.4f, r12 = %.4f, p = %d\n', c, r12, p);
fprintf('det(cov[X1 X2]) = %.6f\n\n', D_src);

fprintf('%-12s %15s %15s %15s\n', 'Measure', 'Theoretical', 'Estimated', 'Abs. error');
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Delta',   delta_th_nonlin,  delta_nonlin,   abs(delta_nonlin   - delta_th_nonlin));
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Delta_A', deltaA_th_nonlin, deltaA_nonlin,  abs(deltaA_nonlin  - deltaA_th_nonlin));
fprintf('%-12s %15.6f %15.6f %15.6f\n', 'Ss',      Ss_th_nonlin,     Ss_est_nonlin,  abs(Ss_est_nonlin - Ss_th_nonlin));

fprintf('\n--- Surrogate significance test for Ss ---\n');
fprintf('Nsurr                    = %d\n', Nsurr);
fprintf('alpha                    = %.4f\n', alpha);
fprintf('Surrogate threshold      = %.6f\n', thr_nonlin);
fprintf('Empirical p-value        = %.6f\n', pval_nonlin);
fprintf('Ss estimated             = %.6f\n', Ss_est_nonlin);

if is_significant_nonlin
    fprintf('Result                   = SIGNIFICANT (Ss > threshold)\n');
else
    fprintf('Result                   = NOT significant (Ss <= threshold)\n');
end

fprintf('\n');

%% =========================================================
%  Plot results
% ==========================================================

figure('Color','w','Position',[100 100 1200 500]);

tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

% Linear system: theoretical vs estimated measures
nexttile;
bar([delta_th_lin delta_lin; ...
     deltaA_th_lin deltaA_lin; ...
     Ss_th_lin Ss_est_lin]);
set(gca,'XTickLabel',{'Delta','Delta_A','S_s'});
ylabel('Value');
title('Linear system');
legend({'Theoretical','Estimated'},'Location','best');
yline(0,'k--');
grid on;

% Linear system: surrogate distribution
nexttile;
histogram(Ss_surr_lin,20);
hold on;
xline(Ss_est_lin,'k','LineWidth',2);
xline(thr_lin,'k--','LineWidth',2);
xlabel('S_s surrogate values');
ylabel('Count');
title('Linear: surrogate test');
legend({'Surrogates','Observed S_s','95% threshold'},'Location','best');
grid on;

% Linear system: source scatter
nexttile;
scatter(X1,X2,15,'filled');
xlabel('X_1');
ylabel('X_2');
title(sprintf('Sources: Spearman r = %.3f', corr(X1,X2,'Type','Spearman')));
grid on;

% Non-linear system: theoretical vs estimated measures
nexttile;
bar([delta_th_nonlin delta_nonlin; ...
     deltaA_th_nonlin deltaA_nonlin; ...
     Ss_th_nonlin Ss_est_nonlin]);
set(gca,'XTickLabel',{'Delta','Delta_A','S_s'});
ylabel('Value');
title('Non-linear multiplicative system');
legend({'Theoretical','Estimated'},'Location','best');
yline(0,'k--');
grid on;

% Non-linear system: surrogate distribution
nexttile;
histogram(Ss_surr_nonlin,20);
hold on;
xline(Ss_est_nonlin,'k','LineWidth',2);
xline(thr_nonlin,'k--','LineWidth',2);
xlabel('S_s surrogate values');
ylabel('Count');
title('Non-linear: surrogate test');
legend({'Surrogates','Observed S_s','95% threshold'},'Location','best');
grid on;

% Non-linear system: Y vs interaction term
nexttile;
scatter(X1.*X2,Y_nonlin,15,'filled');
xlabel('X_1 X_2');
ylabel('Y');
title('Target vs interaction term');
grid on;

sgtitle(sprintf('Theoretical examples: N = %d, r_{12} = %.2f, p = %d', ...
    N, r12, p));

% Optional export
% exportgraphics(gcf,'theoretical_examples_estimation.pdf','ContentType','vector');
% exportgraphics(gcf,'theoretical_examples_estimation.png','Resolution',300);