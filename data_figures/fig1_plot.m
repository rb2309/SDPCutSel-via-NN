function fig1_plot
    nbInstsForEachSize = 30;
    a = csvread('fig1_bounds_3D.csv');
    labels = unique(a(:,1));
    vals = a(:,6);
    valMat = ones(nbInstsForEachSize,size(labels,1))*10;
    for entry=1:size(labels,1)
        valMat(:,entry) = vals((entry-1)*nbInstsForEachSize+1:entry*nbInstsForEachSize);
    end
    % Create figure
    figW = 900;
    figH = 275;
    figure1 = figure('Position', [400,400,figW,figH]);
    % Create axes
    axes1 = axes('Parent',figure1,'Position',...
        [0.25 0.17 0.72 0.75]);
    hold(axes1,'on');
    hold on    
    boxplot(valMat)
    plot(mean(valMat),'LineWidth',4);
    set(axes1,'XTickLabel',labels);
    
    % Create ylabel
    ylabel({'$\frac{z_{qp}(\mathcal{M}+\mathcal{S}^E_3) -z_{qp}(\mathcal{M})}{z_{qp}(\mathcal{M}+\mathcal{S})-z_{qp}(\mathcal{M})}$',' '},...
        'Position',[-2, 0.8, -1],'FontSize',18,...
        'Interpreter','latex',...
        'Rotation',0);

    % Create xlabel
    xlabel('Number of variables ($N$)','FontSize',12,'Interpreter','latex');
    set(axes1,'YTickLabel',...
    {'70%','75%','80%','85%','90%','95%','100%'});
    % Create textbox
    annotation(figure1,'textbox',...
        [0.072673392181589 0.356025839714568 0.0865912301142442 0.183018864775604],...
        'String',{'(\%)'},...
        'LineStyle','none',...
        'Interpreter','latex',...
        'FontSize',18);
    
    
    % Save generated figure    
    set(gcf,'color','w');
    set(gcf, 'renderer','OpenGL');
    saveas(gcf,'fig1_bounds_3D.fig');
    print(gcf,'fig1_bounds_3D','-dpng','-r300');
    close(gcf);
end

