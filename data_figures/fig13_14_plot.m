function fig13_14_plot
    plot_bounds(csvread('fig13_data.csv', 1,2,[1, 2, 12 ,5]), 13);
    plot_bounds(csvread('fig14_data.csv', 1,2,[1, 2, 12 ,4]), 14);
end

function plot_bounds(A, fig_nb)
    figW = 1000;
    figH = 330;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';

    % Create axes
    axes1 = axes('Position',...
        [0.0518819938962359 0.346153846153846 0.911836826965439 0.58152695185114]);
    hold(axes1,'on');
    
    barwidth = 0.6;
    if fig_nb==14
        barwidth = 0.8;
    end
    
    % Create multiple lines using matrix input to bar
    bar1 = bar(A,'BaseValue',100,'EdgeColor','none');
    if fig_nb ==14
        bar1 = bar(A,'BaseValue',0.1,'EdgeColor','none');
    end
    set(bar1(1),'DisplayName','$\mathcal{M}$+$\triangle$','FaceColor',[0 0.5 0],'BarWidth',barwidth);
    set(bar1(2),'DisplayName','Naive $\mathcal{M}$+$\triangle$+$\mathcal{S}_3$',...
        'FaceColor',[0 0 1],...
        'BarWidth',barwidth);
    set(bar1(3),...
        'DisplayName','Heur. $\mathcal{M}$+$\triangle$+$\mathcal{S}_{3-5}$',...
        'FaceColor',[1 0 0],'BarWidth',barwidth);
    if fig_nb==13
        set(bar1(4),'DisplayName','BGL','FaceColor',[0 0 0],'BarWidth',barwidth);
    end
       
    % Set the remaining axes properties
    box(axes1,'on');
    set(axes1,'xlim', [0,13], 'XTick',[1 2 3 4 5 6 7 8 9 10 11 12],'XTickLabel',...
        {'Small Low','Small Medium','Small High','Medium Low','Medium Medium','Medium High','Large Low','Large Medium','Large High','Jumbo Low','Jumbo Medium','Jumbo High'},...
        'XTickLabelRotation',90,'YScale','log');
    h = findobj(gca,'Type','bar');
    if fig_nb==13
        legend1 = legend([h(4),h(3),h(2),h(1)]);
    else
        legend1 = legend([h(3),h(2),h(1)]);
    end
    set(legend1,'Interpreter','latex','FontSize',12,'Location','Northwest');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig%d.fig',fig_nb));
    print(gcf,sprintf('fig%d',fig_nb),'-dpng','-r300');
    close(gcf);
end
