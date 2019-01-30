function fig14_plot
    cut_rounds = 40;
    data_file = 'fig14_data_5cuts.csv';
    bounds = csvread(data_file, 3,0,...
            [3, 0, 2+cut_rounds ,6]);
    cuts_opt = csvread(data_file, 6+cut_rounds,1,...
            [6+cut_rounds, 1, 5+2*cut_rounds ,2]);
    plot_bounds(bounds,5,cut_rounds)
    plot_cuts(cuts_opt,5,cut_rounds)
    
    cut_rounds = 20;
    data_file = 'fig14_data_12cuts.csv';
    bounds = csvread(data_file, 3,0,...
            [3, 0, 2+cut_rounds ,6]);
    cuts_opt = csvread(data_file, 6+cut_rounds,1,...
            [6+cut_rounds, 1, 5+2*cut_rounds ,2]);
    plot_bounds(bounds,12,cut_rounds)
    plot_cuts(cuts_opt,12,cut_rounds)
end

function plot_bounds(A, nb_cuts, cut_rounds)
    rdnText = 'Random sel.';
    comb2Text = '$\mathcal{C}^{(2)}$ Combined selection';
    Feastext = 'Feasibility selection';
    figW = 475;
    figH = 375;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';

    % Create axes
    axes1 = axes('Position',[0.15 0.15 0.80 0.75]);
    hold(axes1,'on');
    % Create multiple lines using matrix input to plot
    plot(A(:,3:7),'DisplayName',rdnText,'LineWidth',2,'LineStyle',':','Color',[0.5 0.5 0.5]);
    plot(A(:,1),'DisplayName',Feastext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[1 0 0],'LineStyle','-');
    plot(A(:,2),'DisplayName',comb2Text,'Marker','square','LineWidth',2,'Color',[0 0 0],'LineStyle','-');

    % Create ylabel
    ylabel('\% of gap from $\mathcal{M}^2$ to optimality closed',...
        'Interpreter','latex');
    % Create xlabel
    xlabel(sprintf('Cut rounds (%d out of 33 cuts selected/round)',nb_cuts) ,...
        'Interpreter','latex');
    % Create title
    title('powerflow0009r','Interpreter','latex');
    
    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'XGrid','on','XMinorGrid','on');
    xlim(axes1,[1 cut_rounds]);
    ylim(axes1,[0 1]);
    
    h = findobj(gca,'Type','line');
    legend1 = legend([h(1),h(2),h(3)]);
    set(legend1,'Interpreter','latex','FontSize',12,'Location','Southeast');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig14_%d_bounds.fig',nb_cuts));
    print(gcf,sprintf('fig14_%d_bounds',nb_cuts),'-dpng','-r300');
    close(gcf);
end

function plot_cuts(A, nb_cuts, cut_rounds)
    comb2Text = '$\mathcal{C}^{(2)}$ Combined selection';
    figW = 475;
    figH = 200;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';

    % Create axes
    axes1 = axes('Position',...
        [0.15 0.20 0.80 0.65]);
    hold(axes1,'on');
    % Create multiple lines using matrix input to plot
    plot(A(:,1),'DisplayName',comb2Text,'Marker','square','LineWidth',2,'Color',[0 0 0],'LineStyle','-');
    
    % Create ylabel
    ylabel({'\# cuts selected by',' optimality measure in $\mathcal{C}^2$'},'FontSize',12,...
        'Interpreter','latex');

    % Create xlabel
    xlabel(sprintf('Cut rounds (%d out of 33 cuts selected/round)',nb_cuts) ,...
        'Interpreter','latex');

    % Create title
    title('powerflow0009r','Interpreter','latex');
    
    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'XGrid','on','XMinorGrid','on');
    xlim(axes1,[1 cut_rounds]);
    ylim(axes1,[0 nb_cuts]);
    
    %legend1 = legend();
    %set(legend1,'Interpreter','latex','FontSize',12,'Location','Northeast');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig14_%d_cuts.fig',nb_cuts));
    print(gcf,sprintf('fig14_%d_cuts',nb_cuts),'-dpng','-r300');
    close(gcf);
end
