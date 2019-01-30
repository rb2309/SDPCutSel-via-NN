% Generate plots for all neural networks (as in fig 7 in the manuscript)
function fig7_plot(varargin)
    % 1 is substracted from nbSamples because indexing starts at 0
    % data files too large for 1M samples
    nbSamples = 500000-1;
    %nbSamples = 10000-1;
    addpath(genpath('..//neural_nets'))
    if ~isempty(varargin)
        plot_net_stats(nbSamples,2,'..//neural_nets//GenDataTest2D_fig7.csv',@neural_net_2D);
        plot_net_stats(nbSamples,3,'..//neural_nets//GenDataTest3D_fig7.csv',@neural_net_3D);
        plot_net_stats(nbSamples,4,'..//neural_nets//GenDataTest4D_fig7.csv',@neural_net_4D);
        plot_net_stats(nbSamples,5,'..//neural_nets//GenDataTest5D_fig7.csv',@neural_net_5D);
    else
        plot_net_stats(nbSamples,2,'..//neural_nets//GenDataTest2D.csv',@neural_net_2D);
        plot_net_stats(nbSamples,3,'..//neural_nets//GenDataTest3D.csv',@neural_net_3D);
        plot_net_stats(nbSamples,4,'..//neural_nets//GenDataTest4D.csv',@neural_net_4D);
        plot_net_stats(nbSamples,5,'..//neural_nets//GenDataTest5D.csv',@neural_net_5D);
    end
end

% Plot regression fit, residuals histogram and residuals vs. fits for each 
% neural net referenced through:
% - the dimension (dim) of the SDP problem it solves
% - the (file) holding the data of the SDP (dim)-dimension sub-problem
%   solved with Mosek
% - the matlab function for the neural net
function plot_net_stats(nbSamples, dim, file, func)
    % skip (plus) columns in .csv file (containing eigenvectors, eigenvalues)
    plus = floor(dim*(dim+3)/2.0);
    nbSamples = min(nbSamples, count_lines(file))-1;  
    X = csvread(file,0,dim*(dim+1),[0,dim*(dim+1),nbSamples,dim*(dim+1)+plus-1]);
    T = csvread(file,0,dim*(dim+1)+plus,[0,dim*(dim+1)+plus,nbSamples,dim*(dim+1)+plus]);
    T = T.';
    X = X.';
    Y = func(X);
    [val,ind] = max(abs(T-Y));
    Y(ind)=[];
    T(ind)=[];
    
    %%%%% Plot regression fit
    h = figure();   
    plotregression(T,Y);
    h.Children(3).Children(1).Color = [0.4843    0.6157    0.9882];
    h.Children(3).Children(2).Color = 'k';
    delete(h.Children(3).Children(3));    
    h2 = figure('Position', [0,0,1000,300]);clf
    subplot(1,3,1);
    ax = gca;
    h.Children(3).Children(1).Parent = h2.Children(1);
    h.Children(3).Children(1).Parent = h2.Children(1);
    legend( h.Children(2).String,'Location', h.Children(2).Location)
    xlabel(h.Children(3).XLabel.String);
    ylabel(h.Children(3).YLabel.String);
    ax.XLim = h.Children(3).XLim;
    ax.YLim = h.Children(3).YLim;   
    close(h)    
    [r,~,~] = regression(T,Y);
    legend({'Data',sprintf('Fit, R^2 = %2.4f%%', r*r*100)},'FontSize',10,'TextColor','black','Location','northwest')    
    xlabel('Output','FontSize',12)
    ylabel('Target','FontSize',12)
    set(ax,'Box','on');     
    
    %%%%% Plot residuals histogram
    figure(h2)
    subplot(1,3,2);
    ax2 = gca;
    hist = histfit(T-Y,50);    
    hist(1).FaceAlpha=0.5; 
    hist(2).Color = [.2 .2 .2];
    pd = fitdist((T-Y)','Normal');
    legend(hist,{'Histogram (10 bins)',sprintf('Normal fit \n ~N(%0.0e, %0.1e)',pd.mu, pd.sigma)},'FontSize',10,'TextColor','black','Location','northeast');   
    xlabel('Residual','FontSize',12)
    ylabel('Probability','FontSize',12)
    % convert to probability
    yt = get(gca, 'YTick');
    set(gca,'YTick',0:(yt(end)/5):yt(end))
    yt = get(gca, 'YTick');
    ytLabels =strings(length(yt),1);
    for i=1:length(yt) 
        ytLabels(i) = sprintf('%.0f%%',yt(i)/nbSamples*100);
    end
    set(gca, 'YTick', yt, 'YTickLabel', ytLabels);
    % rotate
    set(gca,'view',[90 -90]);
    if dim==2
        xlim([-0.015, 0.015])
    elseif dim==3
        xlim([-0.03, 0.03])
    elseif dim==4
        xlim([-0.06, 0.06])
    else
        xlim([-0.08, 0.08])
    end
    
    %%%%% Plot residuals vs. fits
    T = T.';
    Y = Y.';
    [T,I]=sort(T);
    Y = Y(I);
    subplot(1,3,3);
    ax3 = gca;
    fitresult = fit((1:nbSamples)',Y-T,'poly1');
    p11 =  predint(fitresult,(1:nbSamples)',0.95,'observation','off');
    plot(fitresult,(1:nbSamples)',Y-T); hold on, plot((1:nbSamples)',p11,'m--','LineWidth',2,'MarkerFaceColor','k');    
    legend({'Residual point','Fit','95% confidence lines'},'FontSize',10,'TextColor','black','Location','northeast');
    xlabel('Data point','FontSize',12)
    ylabel('Residual','FontSize',12)
    if dim==2
        ylim([-0.08, 0.08])
    elseif dim==3
        ylim([-0.08, 0.08])
    elseif dim==4
        ylim([-0.18, 0.18])
    else
        ylim([-0.22, 0.22])
    end
    xlim([0, nbSamples])
    
    % Change positioning in figure
    pos = get(ax, 'Position');
    pos(1) = pos(1)-0.08;
    set(ax, 'Position', pos)    
    pos = get(ax2, 'Position');
    pos(1) = pos(1)-0.08;
    set(ax2, 'Position', pos)
    pos = get(ax3, 'Position');
    pos(1) = pos(1)-0.08;
    set(ax3, 'Position', pos)
    
    % Save generated figure
    set(gcf,'color','w');
    set(gcf, 'renderer','OpenGL');
    saveas(gcf,sprintf('fig7_NN_%dD.fig',dim));
    print(gcf,sprintf('fig7_NN_%dD',dim),'-dpng','-r300');
    close(gcf);
end
