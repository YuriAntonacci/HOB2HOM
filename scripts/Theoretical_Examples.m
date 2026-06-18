% MatLab Script reproducing theoretical example reported in Section 3 of
% the main paper
%==========================================================================
%                          LINEAR GAUSSIAN SYSTEMS (Section 3.1)
%==========================================================================
% Theoretical trends (Fig. 1)
clear; close all; clc
r12 = [-1:0.01:1];
av=[-1:0.01:1];
bv1 = av; % for Fig. 1a
bv = 0.8*ones(1,length(av)); % Fig. 1b
clims_WMS=[-1 1];
delta=nan*ones(length(av),length(r12)); deltaA=delta;
delta1=nan*ones(length(av),length(r12)); deltaA1=delta1;
for ir=1:length(r12)
    r = r12(ir);
    for ia=1:length(av)
        a = av(ia);
        b = bv(ia);
        if -a^2-b^2-2*a*b*r+1>0 && abs(r)<1 % consistency of covariance matrix
            delta(ia,ir)=-r^2*(a^2+b^2)-2*a*b*r; % Eq. (23)
        end
        b1 = bv1(ia);
        if -a^2-b1^2-2*a*b1*r+1>0 && abs(r)<1 % consistency of covariance matrix
            delta1(ia,ir)=-r^2*(a^2+b1^2)-2*a*b1*r; % Eq. (23)
        end
    end
end
deltaA = delta;
deltaA1 = delta1;
Ss = delta-deltaA; %Eq. (19)
Ss1=delta1-deltaA1; 
%% Plot (Fig.1a-b)
figure;
% Delta, Delta A ed S_s
PAR={'delta','deltaA','Ss'};
PAR1={'delta1','deltaA1','Ss1'};
tit={'\Delta','\Delta_A','S_s'};
for i=1:size(PAR,2)
    % Fig1a
    subplot(3,3,i)
    tmp=eval(PAR{i});
    h=imagesc(flipud(tmp),clims_WMS);
    colormap([bluewhitered]); colorbar;
    set(gca, 'XTick', linspace(1,length(r12),11), 'XTickLabel', linspace(-1,1,11))
    set(gca, 'YTick', linspace(1,length(av),11), 'YTickLabel', linspace(1,-1,11))
    set(h,'alphadata',~isnan(flipud(tmp)));
    set(gca,'color',[0.3 0.3 0.3]);
    xlabel('r_{12}')
    ylabel('a')
    TIT=sprintf('%s [a.u.]',tit{i});
    title(TIT);
    % Fig1b
    subplot(3,3,i+3)
    tmp1=eval(PAR1{i});
    h1=imagesc(flipud(tmp1),clims_WMS);
    colormap([bluewhitered]); colorbar;
    set(gca, 'XTick', linspace(1,length(r12),11), 'XTickLabel', linspace(-1,1,11))
    set(gca, 'YTick', linspace(1,length(av),11), 'YTickLabel', linspace(1,-1,11))
    set(h1,'alphadata',~isnan(flipud(tmp1)));
    set(gca,'color',[0.3 0.3 0.3]);
    xlabel('r_{12}')
    ylabel('a')
    TIT=sprintf('%s [a.u.]',tit{i});
    title(TIT);
end

%==========================================================================
%                         NON-LINEAR SYSTEM (Section 3.2)
%==========================================================================
% Theoretical trends (Fig. 1c)
r12 = [-1:0.01:1];
cv=[-1:0.01:1];
clims_WMS=[-1 1];
delta=nan*ones(length(cv),length(r12)); deltaA=delta; Ss=delta;
for ir=1:length(r12)
    r = r12(ir);
    for ia=1:length(cv)
        c = cv(ia);
    
        if abs(r)<1 % consistency of Covariance matrix
            delta(ia,ir) = c^2*(1-3*r^2); % Eq.(24)
            deltaA(ia,ir) = c^2*(1-3*r^2)-(c^2*(1-r^2)^2/(1+r^2)); % Eq.(25)
            Ss(ia,ir) = (c^2*(1-r^2)^2/(1+r^2)); % Eq.(26)
        end
    end
end

for i=1:size(PAR,2)
    % Fig1c
    subplot(3,3,i+6)
    tmp=eval(PAR{i});
    h=imagesc(flipud(tmp),clims_WMS);
    colormap([bluewhitered]); colorbar;
    set(gca, 'XTick', linspace(1,length(r12),11), 'XTickLabel', linspace(-1,1,11))
    set(gca, 'YTick', linspace(1,length(cv),11), 'YTickLabel', linspace(1,-1,11))
    set(h,'alphadata',~isnan(flipud(tmp)));
    set(gca,'color',[0.3 0.3 0.3]);
    xlabel('r_{12}')
    ylabel('c')
    TIT=sprintf('%s [a.u.]',tit{i});
    title(TIT);
end
