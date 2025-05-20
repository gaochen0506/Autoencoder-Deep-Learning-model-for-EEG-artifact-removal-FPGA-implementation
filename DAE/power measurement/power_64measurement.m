%% 电源电压提取与均值计算
psu_data = load('new64410mesurement.mat', 'Channel1_VDC');
psu_voltage = psu_data.Channel1_VDC;
dt_psu = 0.410;
time_psu = (0:length(psu_voltage)-1)' * dt_psu;

% 设定有效电压范围
v_min = 11.0000;
v_max = 11.8770;

% 提取并计算平均电压
valid_voltage = psu_voltage(psu_voltage >= v_min & psu_voltage <= v_max);
avg_voltage_psu = mean(valid_voltage);
fprintf('PSU Voltage Average (%.4f ~ %.4f V): %.6f V\n', v_min, v_max, avg_voltage_psu);

% 绘图：PSU 电压整体图
figure;
plot(time_psu, psu_voltage, 'k', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize',12);
ylabel('Voltage (V)', 'FontSize',12);
title('PSU Voltage - Full Measurement', 'FontSize',12);
grid on;
set(gca, 'FontSize',12);

%% 功率计算与主图展示
power_data = load('POWER410MEASURE.mat', 'Channel1_VDC');
channel_voltage = power_data.Channel1_VDC;
dt_power = dt_psu;
time_power = (0:length(channel_voltage)-1)' * dt_power;
R = 0.10269;
power = (channel_voltage ./ R) * avg_voltage_psu;

% 绘制整体功率图
figure;
plot(time_power, power, 'k', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize',12);
ylabel('Power (W)', 'FontSize',12);
title('Power Consumption - Full Measurement', 'FontSize',12);
grid on;
set(gca, 'FontSize',12);

%% 子图分段（示意）
% 各分段索引
segment_idx = {
    7:331,
    331:652,
    651:971,
    970:1297,
    1296:1630
};

figure;
colors = ['b', 'g', 'r', 'm', 'c'];
for i = 1:5
    subplot(5,1,i);
    idx = segment_idx{i};
    plot(time_power(idx) - time_power(idx(1)), power(idx), colors(i), 'LineWidth', 2);
    title(sprintf('Repetition %d', i), 'FontSize',12);
    ylabel('Power (W)', 'FontSize',12);
    if i == 5
        xlabel('Time (s)', 'FontSize',12);
    end
    grid on;
    set(gca, 'FontSize',12);
end

%% 分段统计：processing 与 idle
% 时间点（单位秒）
processing_time_sec = [17, 98, 117, 199, 219, 300, 341, 422, 442, 523, ...
                       544, 624, 661, 743, 762, 844, 864, 944, 981, 1061, ...
                       1081, 1162, 1182, 1264, 1306, 1387, 1407, 1489, 1509, 1590];

idle_time_sec = [1, 5.74, 41, 47.56, 82.41, 88.97, 123.82, 138.99, 173.84, ...
                 180.4, 215.66, 222.22, 257.07, 270.19, 305.45, 312.01, ...
                 346.86, 353.42, 388.27, 400.98, 435.83, 442.39, 477.24, ...
                 483.8, 518.65, 534.64, 569.49, 576.46, 611.31, 617.87, ...
                 652.72, 673.63];

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
for i = 3:2:length(idle_idx)-2
    idx_start = idle_idx(i);
    idx_end = idle_idx(i+1);
    avg = mean(power(idx_start:idx_end));
    idle_avgs(end+1) = avg;
    fprintf('Idle interval %d-%d: %.4f W\n', i, i+1, avg);
end

% 输出整体均值
fprintf('\nAverage Processing Power: %.4f W\n', mean(processing_avgs));
fprintf('Average Idle Power: %.4f W\n', mean(idle_avgs));

%% 定义各段索引（重复测量的区间）
segment_idx = {
    7:321,
    331:645,
    651:965,
    970:1284,
    1296:1610
};

% 假设每段数据等长，直接获取每段的索引
numReps = length(segment_idx);
segmentLength = length(segment_idx{1});  % 每段数据点数（所有段等长）
allPower = zeros(numReps, segmentLength);

for i = 1:numReps
    idx = segment_idx{i};
    allPower(i,:) = power(idx);
end

% 计算各时刻的均值与标准差（沿着每个数据点）
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
xlabel('Time (s)', 'FontSize',12);
ylabel('Power (W)', 'FontSize',12);
title('Aggregated Power: Mean & Standard Deviation', 'FontSize',12);
legend('Location','south','Orientation','horizontal','FontSize',12);
grid on;
set(gca, 'FontSize',12);
xlim([0, 130]);
hold off;

width_in = 9;   
height_in = 4;  

% 设置图形属性
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width_in, height_in]);

% 导出为矢量 EMF
print(gcf, '64power_repetitions', '-dmeta');
