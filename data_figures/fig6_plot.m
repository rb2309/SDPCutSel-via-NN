fig1 = figure('Position', [400,400,300,150]);
fig1.Renderer = 'opengl';
x = -4:0.01:4; 
y = tanh(x);
ygrad = 1-tanh(x).^2;
hold on;
plot(x,y,'LineWidth',2,'DisplayName','$\tanh(x)$');
plot(x,ygrad,'LineWidth',2,'DisplayName','$d\tanh/dx$',...
    'Color',[0 0 0],'LineStyle',':');
grid on;
legend1 = legend(gca,'show');
set(legend1,'Location','best','Interpreter', 'latex');
set(fig1, 'renderer','OpenGL');
set(fig1,'color','w');
saveas(fig1,'fig_tanh.fig');
print(fig1,'fig_tanh','-dpng','-r300');
close(fig1);

fig2 = figure('Position', [400,400,300,150]);
fig2.Renderer = 'opengl';
x = -1:0.01:1; 
y = poslin(x);
ygrad = dposlin(x);
hold on;
plot(x,y,'LineWidth',2,'DisplayName','ReLU$(x)$');
plot(x,ygrad,'LineWidth',2,'DisplayName','$d$ReLU$/dx$',...
    'Color',[0 0 0],'LineStyle',':');
grid on;
legend2 = legend(gca,'show');
set(legend2,'Location','best','Interpreter', 'latex');
set(fig2, 'renderer','OpenGL');
set(fig2,'color','w');
saveas(fig2,'fig_relu.fig');
print(fig2,'fig_relu','-dpng','-r300');
close(fig2);
