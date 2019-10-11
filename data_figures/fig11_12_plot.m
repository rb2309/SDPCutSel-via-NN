function fig11_12_plot
    cutsRounds = 20;
    Abounds = csvread('fig11_12_data.csv', 3,0,[3, 0, 3+cutsRounds ,13])*100;
    As = csvread('fig11_12_data.csv', 26,0,[26, 0, 26+cutsRounds ,13])*100;
    Av = csvread('fig11_12_data.csv', 49,0,[49, 0, 49+cutsRounds-1 ,13])*100;
    set(gcf, 'renderer','OpenGL');
    rdnText = 'Random sel.';
    NNtext = 'Optimality sel. (estimated $\hat{\mathcal{I}}_X(\rho)$)';
    EXtext = 'Optimality sel. (exact $\mathcal{I}_X(\rho)$)';
    Feastext = 'Feasibility sel. ($\lambda_{\min}(\rho)$)';
    Combtext = 'Combined sel. ($\mathcal{C}(\rho)$)';
    Boundtext = '$\mathcal{M}+\mathcal{S}^{E}_3$ bound';
    Densetext = 'All violated (up to $N$) dense eigencuts';
    fontSizeLatex =10;
    figW = 375;
    figH = 275;
    
    %%%%%%%%%% First plot both valid and strong in terms of cut rounds/iterations
    %%%% Valid
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';
    axes1 = axes('Position',...
        [0.121621621621622 0.181159420289855 0.836156156156156 0.738840579710145]);
    hold(axes1,'on');
    ytickformat(axes1,'percentage');
    plot1 = plot(Av(:,4:13),'LineWidth',2,'Color',[0.501960813999176 0.501960813999176 0.501960813999176],'LineStyle',':');
    set(plot1(1),'DisplayName',rdnText);
    plot(Av(:,14),'DisplayName',EXtext,'LineWidth',8,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(Av(:,2),'DisplayName',NNtext,'LineWidth',3,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',10);
    plot(Av(:,3),'DisplayName',Feastext,'MarkerSize',10,'Marker','square','LineWidth',3,'Color',[1 0 0],'LineStyle','-');
    plot(Av(:,1),'DisplayName',Combtext,'LineWidth',4,'Color',[0 0 0],'LineStyle','-');
    xlim(axes1,[1 20]);
    box(axes1,'on');
    set(axes1,'FontSize',10,'XGrid','on');
    xlabel('Cut rounds','Interpreter','latex','FontSize',fontSizeLatex+4);
    set(gcf,'color','w');
    saveas(gcf,'fig12_rounds.fig');
    print(gcf,'fig12_rounds','-dpng','-r300');
    
    % save common legend
    h = findobj(gca,'Type','line');
    legend1 = legend([h(3),h(4),h(2),h(1),h(14)]);
    set(legend1,'Location','southwest','Interpreter','latex','FontSize',fontSizeLatex+6);
    legend1.EdgeColor = 'white';
    saveLegendToImage(fig1, legend1);
    close(gcf);
    
    %%%% Strong
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';
    axes1 = axes('Position',...
        [0.121621621621622 0.181159420289855 0.836156156156156 0.738840579710145]);
    hold(axes1,'on');
    ytickformat(axes1,'percentage');
    plot1 = plot(As(:,4:13),'LineWidth',2,'Color',[0.501960813999176 0.501960813999176 0.501960813999176],'LineStyle',':');
    set(plot1(1),'DisplayName',rdnText);
    plot(As(:,14),'DisplayName',EXtext,'LineWidth',8,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(As(:,2),'DisplayName',NNtext,'LineWidth',3,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',10);
    plot(As(:,3),'DisplayName',Feastext,'MarkerSize',10,'Marker','square','LineWidth',3,'Color',[1 0 0],'LineStyle','-');
    plot(As(:,1),'DisplayName',Combtext,'LineWidth',4,'Color',[0 0 0],'LineStyle','-');
    xlim(axes1,[1 20]);
    box(axes1,'on');
    set(axes1,'FontSize',10,'XGrid','on');
    xlabel('Cut rounds','Interpreter','latex','FontSize',fontSizeLatex+4);
    set(gcf,'color','w');
    saveas(gcf,'fig11_rounds.fig');
    print(gcf,'fig11_rounds','-dpng','-r300');
    close(gcf);
    
    % %%%%%%%%%% Second, plot both valid and strong in terms of bounds
    %%%% Valid
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';
    axes1 = axes('Position',...
        [0.121621621621622 0.181159420289855 0.836156156156156 0.738840579710145]);
    hold(axes1,'on');
    ytickformat(axes1,'percentage');
    xtickformat(axes1,'percentage');
    Abounds2 = Abounds(1:end-1,:);
    plot1 = plot(Abounds2(:,4:13),Av(:,4:13),'LineWidth',2,'Color',[0.501960813999176 0.501960813999176 0.501960813999176],'LineStyle',':');
    set(plot1(1),'DisplayName',rdnText);
    plot(Abounds2(:,14),Av(:,14),'DisplayName',EXtext,'LineWidth',8,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(Abounds2(:,2),Av(:,2),'DisplayName',NNtext,'LineWidth',3,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',10);
    plot(Abounds2(:,3),Av(:,3),'DisplayName',Feastext,'MarkerSize',10,'Marker','square','LineWidth',3,'Color',[1 0 0],'LineStyle','-');
    plot(Abounds2(:,1),Av(:,1),'DisplayName',Combtext,'LineWidth',4,'Color',[0 0 0],'LineStyle','-');
    xlim(axes1,[0 80]);
    box(axes1,'on');
    set(axes1,'FontSize',10,'XGrid','on');
    xlabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed','Interpreter','latex','FontSize',fontSizeLatex+4);
    set(gcf,'color','w');
    saveas(gcf,'fig12_bounds.fig');
    print(gcf,'fig12_bounds','-dpng','-r300');
    close(gcf);
    
    %%%% Strong
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';
    axes1 = axes('Position',...
        [0.121621621621622 0.181159420289855 0.836156156156156 0.738840579710145]);
    hold(axes1,'on');
    ytickformat(axes1,'percentage');
    xtickformat(axes1,'percentage');

    plot1 = plot(Abounds(:,4:13),As(:,4:13),'LineWidth',2,'Color',[0.501960813999176 0.501960813999176 0.501960813999176],'LineStyle',':');
    set(plot1(1),'DisplayName',rdnText);
    plot(Abounds(:,14),As(:,14),'DisplayName',EXtext,'LineWidth',8,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(Abounds(:,2),As(:,2),'DisplayName',NNtext,'LineWidth',3,'Color',[0 0 1],'LineStyle','-','Marker','o', 'MarkerSize',10);
    plot(Abounds(:,3),As(:,3),'DisplayName',Feastext,'MarkerSize',10,'Marker','square','LineWidth',3,'Color',[1 0 0],'LineStyle','-');
    plot(Abounds(:,1),As(:,1),'DisplayName',Combtext,'LineWidth',4,'Color',[0 0 0],'LineStyle','-');
    xlim(axes1,[0 80]);
    box(axes1,'on');
    set(axes1,'FontSize',10,'XGrid','on');
    xlabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed','Interpreter','latex','FontSize',fontSizeLatex+4);
    set(gcf,'color','w');
    saveas(gcf,'fig11_bounds.fig');
    print(gcf,'fig11_bounds','-dpng','-r300');
    close(gcf);
end

function saveLegendToImage(figHandle, legHandle)
    warning('off','all');
    %make all contents in figure invisible
    allLineHandles = findall(figHandle, 'type', 'line');
    for i = 1:length(allLineHandles)
        allLineHandles(i).XData = NaN; %ignore warnings
    end
    %make axes invisible
    axis off
    %move legend to lower left corner of figure window
    legHandle.Units = 'pixels';
    boxLineWidth = legHandle.LineWidth;
    %save isn't accurate and would swallow part of the box without factors
    legHandle.Position = [6 * boxLineWidth, 6 * boxLineWidth, ...
        legHandle.Position(3), legHandle.Position(4)];
    legLocPixels = legHandle.Position;
    %make figure window fit legend
    figHandle.Units = 'pixels';
    figHandle.InnerPosition = [1, 1, legLocPixels(3) + 12 * boxLineWidth, ...
        legLocPixels(4) + 12 * boxLineWidth];
    %save legend
    saveas(figHandle,'fig11_legend.fig');
    print(figHandle,'fig11_legend','-dpng','-r300');
    warning('on','all');
    close(gcf);
end
