% Validation in a stochastic autoregressive system (Section 4)
%
% This script simulates the stochastic autoregressive system used to validate
% the proposed framework in a controlled dynamical setting.
%
% The system contains:
%   - two source processes, X1 and X2;
%   - one target process, Y;
%   - an additive source contribution, controlled by cA;
%   - a non-additive source interaction, controlled by c;
%   - source dependence, controlled by r12.
%
% The chosen setting, with strong source dependence and a non-additive
% interaction term, is expected to reproduce a masked structural synergy
% regime: Delta can become negative because of redundancy, while Ss remains
% positive because the target still contains a non-additive predictive
% component.

clear all; close all; clc;

%% Parameters

N = 1000;          % length of the simulated time series

c  = 2;            % strength of the non-additive interaction X1*X2
cA = 0.05;         % strength of the additive source effects

r12 = -0.8;        % coupling from X1 to X2, inducing source dependence

p = 2;             % maximum polynomial order used in the regression models

Nsurr = 100;       % number of surrogate realizations
alpha = 0.05;      % significance level

rng(1);            % reproducibility

%% Simulate the stochastic autoregressive system
% Equations corresponding to Eq. 27a-c in the manuscript:
%
% X1_n = 0.8 X1_{n-1} + 0.2 e1_n
% X2_n = 0.6 X2_{n-1} + r12 X1_{n-1} + 0.2 e2_n
% Y_n  = 0.5 Y_{n-1} + c X1_{n-1} X2_{n-1}
%        + cA X1_{n-1} + cA X2_{n-1} + 0.2 e3_n
%
% The target depends on the lagged source states, so the analysis below
% uses Y_n as target and X1_{n-1}, X2_{n-1} as sources.

x0 = -1 + rand(1,3);

X1 = zeros(N,1);
X2 = zeros(N,1);
Y  = zeros(N,1);

X1(1) = x0(1);
X2(1) = x0(2);
Y(1)  = x0(3);

for t = 2:N

    X1(t) = 0.8 * X1(t-1) + 0.2 * randn;

    X2(t) = 0.6 * X2(t-1) + r12 * X1(t-1) + 0.2 * randn;

    Y(t) = 0.5 * Y(t-1) ...
         + c  * (X1(t-1) * X2(t-1)) ...
         + cA * X1(t-1) ...
         + cA * X2(t-1) ...
         + 0.2 * randn;

end

%% Build target and source variables for the analysis
% The current value of Y is used as target.
% The lagged values of X1 and X2 are used as sources.

y  = Y(2:end);
x1 = X1(1:end-1);
x2 = X2(1:end-1);

% Empirical source-source correlation.
% Note that this is not necessarily equal to the parameter r12, because r12
% is the coupling coefficient in the autoregressive equation, not directly
% the sample correlation coefficient.
r_emp = corr(x1, x2, "Type", "Spearman");

%% Compute Delta, Delta_A, and structural synergy Ss

% Structural synergy:
% surr_sign = 0 -> original data, no surrogate shuffling
% surr_p    = 0 -> no order-p surrogate shuffling
Ss = struct_syn(x1, x2, y, p, 0, 0);

% Interaction predictability and additive interaction predictability
[delta, deltaA] = int_pred(x1, x2, y, p);

%% Surrogate analysis for the statistical significance of Ss
% The null hypothesis is that the interaction terms do not provide an
% additional non-additive predictive contribution to the target.
%
% surr_sign = 1 randomly shuffles the interaction terms, disrupting their
% correspondence with the target while preserving the additive source terms.

Ss_surr = zeros(Nsurr,1);

for j = 1:Nsurr
    Ss_surr(j) = struct_syn(x1, x2, y, p, 1, 0);
end

% Empirical upper-tail p-value.
% The +1 correction avoids p-values equal to zero with finite surrogates.
pval = (sum(Ss_surr >= Ss) + 1) / (Nsurr + 1);

% Percentile-based threshold, as described in the manuscript
thr = prctile(Ss_surr, 100*(1-alpha));

% Significance decision
is_significant = Ss > thr;

%% Print results

fprintf('\n=== Stochastic autoregressive system ===\n');
fprintf('Parameters: c = %.4f, cA = %.4f, r12 = %.4f, p = %d\n', c, cA, r12, p);
fprintf('N = %d, Nsurr = %d, alpha = %.4f\n', N, Nsurr, alpha);
fprintf('Empirical Spearman corr(x1,x2) = %.4f\n\n', r_emp);

fprintf('%-12s %15s\n', 'Measure', 'Estimated');
fprintf('%-12s %15.6f\n', 'Delta',   delta);
fprintf('%-12s %15.6f\n', 'Delta_A', deltaA);
fprintf('%-12s %15.6f\n', 'Ss',      Ss);

fprintf('\n--- Surrogate significance test for Ss ---\n');
fprintf('Surrogate threshold      = %.6f\n', thr);
fprintf('Empirical p-value        = %.6f\n', pval);
fprintf('Ss estimated             = %.6f\n', Ss);

if is_significant
    fprintf('Result                   = SIGNIFICANT (Ss > threshold)\n');
else
    fprintf('Result                   = NOT significant (Ss <= threshold)\n');
end

fprintf('\n');

%% Plot results

figure('Color','w','Position',[100 100 1100 450]);

tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% Panel 1: estimated measures
nexttile;
bar([delta deltaA Ss]);
set(gca,'XTickLabel',{'Delta','Delta_A','S_s'});
ylabel('Value');
title('Estimated measures');
yline(0,'k--','LineWidth',1);
grid on;

% Panel 2: surrogate distribution for Ss
nexttile;
histogram(Ss_surr,20);
hold on;
xline(Ss,'k','LineWidth',2);
xline(thr,'k--','LineWidth',2);
xlabel('S_s surrogate values');
ylabel('Count');
title('Surrogate distribution');
legend({'Surrogates','Observed S_s','95% threshold'},'Location','best');
grid on;

% Panel 3: source dependence
nexttile;
scatter(x1,x2,15,'filled');
xlabel('X_1 lagged');
ylabel('X_2 lagged');
title(sprintf('Source dependence: Spearman r = %.3f', r_emp));
grid on;

sgtitle(sprintf('AR validation: c = %.2f, c_A = %.2f, r_{12} = %.2f, p = %d', ...
    c, cA, r12, p));
