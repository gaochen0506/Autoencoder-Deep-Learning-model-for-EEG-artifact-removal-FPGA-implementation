% 电阻测量数据（修改这里的数据即可）
R = [0.1017, 0.1054, 0.1017, 0.1044, 0.1047, ...
     0.0985, 0.0991, 0.0999, 0.1005, 0.1052, ...
     0.1051, 0.1002, 0.1024, 0.1056, 0.1060];

% 样本个数
n = length(R);

% 平均值
R_mean = mean(R);

% 标准差（样本标准差）
R_std = std(R, 1);         % 总体标准差
R_std_sample = std(R, 0);  % 样本标准差（推荐用于置信区间）

% 标准误差（standard error）
SE = R_std_sample / sqrt(n);

% 置信区间（95%，双侧），使用t分布
alpha = 0.05;
t_value = tinv(1 - alpha/2, n - 1);
CI_lower = R_mean - t_value * SE;
CI_upper = R_mean + t_value * SE;

% 显示结果
fprintf('平均值: %.5f Ω\n', R_mean);
fprintf('标准差: %.5f Ω\n', R_std_sample);
fprintf('标准误差: %.5f Ω\n', SE);
fprintf('95%% 置信区间: [%.5f, %.5f] Ω\n', CI_lower, CI_upper);
