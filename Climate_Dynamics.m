% Script - Climate case study (Sect. 5.1)
% This script evaluates interaction predictability (Delta),
% additive interaction predictability (Delta_A), and structural synergy (S_s)
% for different climate-index triplets.
%
% The target variable is fixed, while different pairs of climate indices
% are used as sources. For each source pair and polynomial order p,
% the script computes:
%   - source-source Spearman correlation
%   - Delta
%   - Delta_A
%   - S_s = Delta - Delta_A
%   - surrogate-based p-values for S_s
%   - surrogate-based p-values for the increase in S_s due to order-p terms

clear all; close all; clc

% Load climate data matrix.
% Rows are time samples, columns are climate indices.
load("climate_data.mat");

% Number of surrogate realizations
Nsurr = 100;

% Index of the target variable.
% In the paper this corresponds to SOI.
indY = 7;

% Indices of the source pairs.
% Each element of indX1 and indX2 defines one source pair.
% Example: pair is {clima(:,indX1(is)), clima(:,indX2(is))}
indX1 = [11 11 11 8 8 10];
indX2 = [8 10 12 10 12 12];

% Polynomial orders tested for the regression models
p = 2:4;

% Preallocate p-value matrices:
% rows = source pairs, columns = polynomial orders
pval_ss = nan(length(indX1), length(p));   % p-values for S_s significance
pval_p  = nan(length(indX1), length(p));   % p-values for order-p improvement

% Preallocate measure matrices
corrl  = nan(length(indX1), 1);
Ss     = nan(length(indX1), length(p));
delta  = nan(length(indX1), length(p));
deltaA = nan(length(indX1), length(p));

% Loop over source pairs
for is = 1:length(indX1)

    % Select target and source time series
    y  = clima(:, indY);        % target climate index, e.g. SOI
    x1 = clima(:, indX1(is));   % first source climate index
    x2 = clima(:, indX2(is));   % second source climate index

    % Source-source dependence, used to interpret redundancy/synergy effects
    corrl(is,1) = corr(x1, x2, "Type", "Spearman");

    % Loop over polynomial orders
    for ip = 1:length(p)

        % Current polynomial order
        pp = p(ip);

        % Structural synergy estimated from original data
        % surr_sign = 0: no surrogate shuffling
        % surr_p    = 0: no order-p surrogate shuffling
        ss = struct_syn(x1, x2, y, pp, 0, 0);
        Ss(is, ip) = ss;

        % Interaction predictability and additive interaction predictability
        [delta(is,ip), deltaA(is,ip)] = int_pred(x1, x2, y, pp);

        % -------------------------------------------------------------
        % Surrogate test 1:
        % Test whether S_s is larger than expected under the null
        % hypothesis of no non-additive predictive contribution.
        %
        % surr_sign = 1 shuffles the interaction terms, disrupting
        % their relation with the target while preserving the additive terms.
        % -------------------------------------------------------------
        Ss_surr = zeros(Nsurr,1);

        for j = 1:Nsurr
            Ss_surr(j) = struct_syn(x1, x2, y, pp, 1, 0);
        end

        % Empirical upper-tail p-value.
        % The +1 correction avoids zero p-values with finite surrogates.
        pval_ss(is,ip) = (sum(Ss_surr >= ss) + 1) / (Nsurr + 1);

        % -------------------------------------------------------------
        % Surrogate test 2:
        % Test whether the terms introduced at polynomial order pp
        % provide a significant increase in structural synergy beyond
        % lower-order terms.
        %
        % surr_p = 1 shuffles only the newly introduced order-p terms,
        % while keeping lower-order terms unchanged.
        % -------------------------------------------------------------
        Ss_surr_p = zeros(Nsurr,1);

        for j = 1:Nsurr
            Ss_surr_p(j) = struct_syn(x1, x2, y, pp, 0, 1);
        end

        % Empirical upper-tail p-value for order-p contribution
        pval_p(is,ip) = (sum(Ss_surr_p >= ss) + 1) / (Nsurr + 1);

    end

end

% Significance masks
sign_ss = (pval_ss < 0.05);   % significant S_s
sign_p  = (pval_p  < 0.05);   % significant contribution of order-p terms

% Colors used in the figure
col = [88,129,87; ...
       229,75,75; ...
       242,175,41; ...
       71,71,71] / 255;

% -------------------------------------------------------------
% Plot results
% -------------------------------------------------------------
figure;

subplot(4,1,1);
bar(corrl, 'FaceColor', col(1,:));
ylim([-0.1 0.5]);
ylabel('corr');
title('Source-source Spearman correlation');

subplot(4,1,2);
bar(delta, 'FaceColor', col(2,:));
ylim([-0.18 0.03]);
ylabel('\Delta');
title('Interaction predictability');

subplot(4,1,3);
bar(deltaA, 'FaceColor', col(3,:));
ylim([-0.18 0.03]);
ylabel('\Delta_A');
title('Additive interaction predictability');

subplot(4,1,4);
bar(Ss, 'FaceColor', col(4,:));
ylim([-0.001 0.015]);
ylabel('S_s');
xlabel('Source pair');
title('Structural synergy');

% Optional: add legend for polynomial orders
legend(arrayfun(@(x) sprintf('p = %d', x), p, 'UniformOutput', false), ...
       'Location', 'best');