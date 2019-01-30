function fig9_plot
    cutsRounds = 20;
    fid = fopen('fig9_data.csv');
    nb_instances = floor(count_lines('fig9_data.csv')/25);   
    for figNb=1:nb_instances
        row1 = (cutsRounds+5)*(figNb-1)+2;
        fseek(fid,0,'bof');
        a = textscan(fid,'%s','Delimiter',',','headerlines',row1-2);
        filename = a{1}{1};
        b = csvread('fig9_data.csv', row1,0,[row1, 0, row1 ,2]);
        A = csvread('fig9_data.csv', row1+2,0,...
            [row1+2, 0, row1+2+cutsRounds ,13]);
        plotInstance(A,b,figNb,filename)
    end    
end

function plotInstance(A, b,figNb,filename)
    rdnText = 'Random sel.';
    NNtext = 'Optimality sel. (estimated $\hat{\mathcal{I}}_X(\rho)$)';
    EXtext = 'Optimality sel. (exact $\mathcal{I}_X(\rho)$)';
    Feastext = 'Feasibility sel.';
    Boundtext = '$\mathcal{M}+\mathcal{S}_3$';
    figW = 475;
    figH = 375;
    fig1 = figure('Position', [400,400,figW,figH]);
    fig1.Renderer = 'opengl';

    % Create axes
    axes1 = axes('Position',...
        [0.151927437641723 0.160278745644599 0.811791383219955 0.707270923229573]);
    hold(axes1,'on');
    % Create multiple lines using matrix input to plot
    plot1 = plot(A(:,3:12),'LineWidth',2,'LineStyle',':','Color',[0.5 0.5 0.5]);
    set(plot1(1),'DisplayName',rdnText);
    plot(A(:,13),'DisplayName',EXtext,'LineWidth',6,'LineStyle',':','Color',[0.301960784313725 0.749019607843137 0.929411764705882]);
    plot(A(:,1),'DisplayName',NNtext,'LineWidth',2,'Color',[0 0 1],'LineStyle','-','Marker','square', 'MarkerSize',6);
    plot(A(:,2),'DisplayName',Feastext,'MarkerSize',6,'Marker','square','LineWidth',2,'Color',[1 0 0],'LineStyle','-');
    plot(A(:,14),'DisplayName',Boundtext,'LineWidth',2,'Color',[0 0.5 0],'LineStyle','-');

    % Create ylabel
    ylabel('\% of $\mathcal{M}$ to $(\mathcal{M}+\mathcal{S})$ gap closed',...
        'FontSize',12,...
        'Interpreter','latex');
    % Create xlabel
    xlabel(['Cut rounds (5\% $\rho$ selected, max ', sprintf('%d cuts/round)', b(3))],'FontSize',12,...
        'Interpreter','latex');
    % Create title
    title([filename,' (' , num2str(b(1)),' vars, ' , num2str(b(2)), '\% dense)'],...
        'FontSize',14,'FontWeight','normal',...
        'Interpreter','latex');

    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'XGrid','on','XMinorGrid','on');
    xlim(axes1,[1 20]);
    ylim(axes1,[0 1]);
    
    h = findobj(gca,'Type','line');
    legend1 = legend([h(3),h(4),h(2),h(14),h(1)]);
    set(legend1,'Interpreter','latex','FontSize',12,'Location','best');
    set(gcf, 'renderer','OpenGL');
    set(gcf,'color','w');
    saveas(gcf,sprintf('fig9_%d.fig',figNb));
    print(gcf,sprintf('fig9_%d',figNb),'-dpng','-r300');
    close(gcf);
end
