function fig11_plot
    cutsRounds = 20;
    % read bounds data
    A = csvread('fig10-11_data.csv', 3,0,[3, 0, 3+cutsRounds ,13]);
    figW = 475;
    figH = 375;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';
    axes1 = axes('Position',...
        [0.151927437641723 0.160278745644599 0.811791383219955 0.707270923229573]);
    hold(axes1,'on');
    % Create plot
    plot1 = loglog(20-[0:1:20],1-A(:,4:13),'LineWidth',2,...
        'Color',[0.501960813999176 0.501960813999176 0.501960813999176],...
        'LineStyle',':');
    set(plot1(1),'DisplayName','Random sel.');
    loglog(20-[0:1:20],1-A(:,2),'DisplayName','Optimality sel. (estimated $\hat{\mathcal{I}}_X(\rho)$)','LineWidth',2,'Color',[0 0 1],'LineStyle','-','Marker','square');
    loglog(20-[0:1:20],1-A(:,3),'DisplayName','Feasibility sel. (exact $\mathcal{I}_X(\rho)$)','LineWidth',2,'Color',[1 0 0],'LineStyle','-','Marker','square');
    loglog(20-[0:1:20],1-A(:,1),'DisplayName','Combined sel. ($\mathcal{C}(\rho)$)','LineWidth',2,'Color',[0 0 0],'LineStyle','-','Marker','o', 'MarkerSize',8);
    fontSizeLatex =10;
    % Create ylabel
    ylabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed','Interpreter','latex');
    % Create xlabel
    xlabel('Cut rounds (5\% $\rho$ selected, max 115 cuts/round)','Interpreter','latex');
    % Create title
    title('spar-100-025-1 (100 vars, 25\% dense)','FontSize',fontSizeLatex,'Interpreter','latex');
    xlim(axes1,[1 20]);
    ylim(axes1,[0 1]);
    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'FontSize',10,'XGrid','on','XMinorGrid','on');

    % Log scales
    set(gca,'ydir','reverse','yscale','log')
    set(axes1,'XGrid','on','XMinorGrid','on','XDir','reverse','XMinorTick','on',...
        'XScale','log','XTickLabel',{'20','10'},'YDir','reverse','YMinorTick','on','YScale','log',...
        'YTick',[0 0.2 0.4 0.6 0.8 1],'YTickLabel',{'1','0.8','0.6','0.4','0.2','0'});
    set(gca,'xscale','log')
    % Legend
    h = findobj(gca,'Type','line');
    legend1 = legend([h(1),h(3),h(2),h(13)]);
    set(legend1,'Interpreter','latex','FontSize',fontSizeLatex,'Location','best');
    set(gcf,'color','w');
    saveas(gcf,'fig11.fig');
    print(gcf,'fig11','-dpng','-r300');
    close(gcf);
end
