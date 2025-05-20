%% 电源电压提取与均值计算
psu_data = load('psu128_411.mat', 'Channel1_VDC');
psu_voltage = psu_data.Channel1_VDC;
dt_psu = 0.411;
time_psu = (0:length(psu_voltage)-1)' * dt_psu;

% 设定有效电压范围
v_min = 11.9720;
v_max = 11.9760;

% 提取并计算平均电压
valid_voltage = psu_voltage(psu_voltage >= v_min & psu_voltage <= v_max);
avg_voltage_psu = mean(valid_voltage);
fprintf('PSU Voltage Average (%.4f ~ %.4f V): %.6f V\n', v_min, v_max, avg_voltage_psu);

% 绘图：PSU 电压整体图
figure;
plot(time_psu, psu_voltage, 'k', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Voltage (V)', 'FontSize', 12);
title('PSU Voltage - Full Measurement', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

%% 功率计算与主图展示
power_data = load('power128_411.mat', 'Channel1_VDC');
channel_voltage = power_data.Channel1_VDC;
dt_power = dt_psu;
time_power = (0:length(channel_voltage)-1)' * dt_power;
R = 0.10269;
power = (channel_voltage ./ R) * avg_voltage_psu;

% 绘制整体功率图
figure;
plot(time_power, power, 'k', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Power (W)', 'FontSize', 12);
title('Power Consumption - Full Measurement', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

%% 子图分段（示意）
% 各分段索引
segment_idx = {
    25:998,
    1011:1984,
    2035:3008,
    3008:3981,
    4036:5009
};

figure;
colors = ['b', 'g', 'r', 'm', 'c'];
for i = 1:5
    subplot(5,1,i);
    idx = segment_idx{i};
    plot(time_power(idx) - time_power(idx(1)), power(idx), colors(i), 'LineWidth', 2);
    title(sprintf('Repetition %d', i), 'FontSize', 12);
    ylabel('Power (W)', 'FontSize', 12);
    if i == 5
        xlabel('Time (s)', 'FontSize', 12);
    end
    grid on;
    set(gca, 'FontSize', 12);
end

%% 分段统计：processing 与 idle
% 时间点（单位秒）
processing_time_sec = [17, 98, 117, 199, 219, 300, 341, 422, 442, 523, ...
                       544, 624, 661, 743, 762, 844, 864, 944, 981, 1061, ...
                       1081, 1162, 1182, 1264, 1306, 1387, 1407, 1489, 1509, 1590];

idle_time_sec = [1, 13.563, 53.019, 59.184, 99.051, 105.216, 144.672, 150.837, 190.704, ...
                 196.869, 236.736, 242.901, 282.768, 288.933, 328.389, 334.554, ...
                 374.421, 380.586, 403.191, 418.809, 458.676, 464.841, 504.297, ...
                 510.462, 550.329, 556.494, 596.361, 602.115, 641.982, 648.147, ...
                 688.014, 694.179, 733.635, 739.8, 779.667, 785.421, 808.026, 839.673, ...
                 885.705, 897.54, 925.161, 931.326, 971.193, 977.358, 1016.91, 1022.98, ...
                 1062.85, 1069.01, 1108.88, 1115.04, 1154.5, 1160.66, 1200.53, 1206.28, ...
                 1228.89, 1239.58, 1279.44, 1285.20, 1325.06, 1330.82, 1370.69, 1376.85, ...
                 1416.31, 1422.47, 1461.93, 1468.09, 1507.55, 1513.71, 1553.58, 1559.33, ...
                 1599.20, 1604.95, 1627.56, 1662.08, 1701.54, 1707.70, 1747.16, 1753.33, ...
                 1793.19, 1798.95, 1838.81, 1844.98, 1884.43, 1890.60, 1930.06, 1936.22, ...
                 1976.09, 1981.84, 2021.71, 2027.46, 2050.07, 2093.22];

% 转换为采样点索引
processing_idx = round(processing_time_sec);
idle_idx = round(idle_time_sec / dt_power);

% 处理态功率均值计算（每两个点为一个区间）
processing_avgs = [];
for i = 1:2:length(processing_idx)
    idx_start = processing_idx(i);
    idx_end = processing_idx(i+1);
    avg = mean(power(idx_start:idx_end));
    processing_avgs(end+1) = avg;
    fprintf('Processing interval %d-%d: %.4f W\n', i, i+1, avg);
end

% 空闲态功率均值计算（每两个点为一个区间）
idle_avgs = [];
for i = 1:2:length(idle_idx)
    idx_start = idle_idx(i);
    idx_end = idle_idx(i+1);
    avg = mean(power(idx_start:idx_end));
    idle_avgs(end+1) = avg;
    fprintf('Idle interval %d-%d: %.4f W\n', i, i+1, avg);
end

% 输出整体均值
fprintf('\nAverage Processing Power: %.4f W\n', mean(processing_avgs));
fprintf('Average Idle Power: %.4f W\n', mean(idle_avgs));

%% 数据分段整合：构建 allPower 矩阵（假设各段数据等长）
numReps = length(segment_idx);
segmentLength = length(segment_idx{1});  % 每段数据点数（所有段等长）
allPower = zeros(numReps, segmentLength);

for i = 1:numReps
    idx = segment_idx{i};
    allPower(i,:) = power(idx);
end

%% 计算各时刻的均值与标准差（沿着每个数据点）
meanPower = mean(allPower, 1);
stdPower = std(allPower, 0, 1);

% 取其中一段的时间作为横轴（假设各段的时间间隔一致）
timeSegment = time_power(segment_idx{1}) - time_power(segment_idx{1}(1));

% 绘图：均值和标准差（均为实线，线宽1）
figure;
plot(timeSegment, meanPower, 'r-', 'LineWidth', 1, 'DisplayName', 'Mean Power');
hold on;
plot(timeSegment, meanPower + stdPower, 'b-', 'LineWidth', 1, 'DisplayName', 'Mean + STD');
plot(timeSegment, meanPower - stdPower, 'b-', 'LineWidth', 1, 'DisplayName', 'Mean - STD');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Power (W)', 'FontSize', 12);
title('Aggregated Power: Mean & Standard Deviation', 'FontSize', 12);
legend('Location','south','Orientation','horizontal','FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
hold off;

%% 新增部分：基于阈值计算运行态（功率 > 13.36 W）与空闲态（功率 < 12.9 W）的统计量
running_values = allPower(allPower > 13.36);
idle_values = allPower(allPower < 12.9);

mean_running = mean(running_values);
std_running = std(running_values);

mean_idle = mean(idle_values);
std_idle = std(idle_values);

fprintf('运行态（功率 > 13.36 W）: 平均值 = %.4f W, 标准差 = %.4f W\n', mean_running, std_running);
fprintf('空闲态（功率 < 12.9 W）: 平均值 = %.4f W, 标准差 = %.4f W\n', mean_idle, std_idle);

width_in = 9;   
height_in = 4;  

% 设置图形属性
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width_in, height_in]);

% 导出为矢量 EMF
print(gcf, '128power_repetitions', '-dmeta');
