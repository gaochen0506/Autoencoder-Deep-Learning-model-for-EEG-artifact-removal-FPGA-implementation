final_result = load('final_result.mat','outputfromH');
clean_result = load('x_test_clean1.mat','data_to_show');
noisy_result = load('x_test_noisy1.mat','data_to_test');

%creat matrix   
x_test_noisy = single(noisy_result.data_to_test);   % noisy signal
x_test_clean = single(clean_result.data_to_show);   % clean signal
decoded_layer = single(final_result.outputfromH');   % denoisy signal


z_test_noisy    = x_test_noisy - mean(x_test_noisy, 2);
z_test_clean    = x_test_clean - mean(x_test_clean, 2);
z_decoded_layer = decoded_layer - mean(decoded_layer, 2);

numSignals = size(z_test_clean, 1);
CC_detectClean = arrayfun(@(i) corrcoef(z_test_clean(i, :), z_test_noisy(i, :)), 1:numSignals, 'UniformOutput', false);
% 提取相关系数
CC_detectClean = cellfun(@(r) r(1, 2), CC_detectClean);

% 根据相关系数划分信号
clean_detect = find(CC_detectClean > 0.95);
noisy_detect = find(CC_detectClean <= 0.95);

% 初始化 cell 数组

num_EOG = 0;
num_Motion = 0;
num_EMG = 0;


clean_inputs = zeros(length(clean_detect), 800);
clean_outputs = zeros(length(clean_detect), 800);

for i = 1:length(clean_detect)
    idx = clean_detect(i);
    clean_inputs(i,:)  = z_test_noisy(idx, :); 
    clean_outputs(i,:) = z_decoded_layer(idx, :);      
end

for i = 1:length(noisy_detect)
    idx = noisy_detect(i);
    if idx < 345
        num_EOG = num_EOG + 1;
    elseif idx >= 345 && idx < 967
        num_Motion = num_Motion + 1;
    elseif idx >= 967
        num_EMG = num_EMG + 1;
    end
end


num_EOG = 0;
num_Motion = 0;
num_EMG = 0;

clean_inputs = zeros(length(clean_detect), 800);
clean_outputs = zeros(length(clean_detect), 800);

for i = 1:length(clean_detect)
    idx = clean_detect(i);
    clean_inputs(i,:)  = z_test_noisy(idx, :); 
    clean_outputs(i,:) = z_decoded_layer(idx, :);      
end

for i = 1:length(noisy_detect)
    idx = noisy_detect(i);
    if idx < 345
        num_EOG = num_EOG + 1;
    elseif idx >= 345 && idx < 967
        num_Motion = num_Motion + 1;
    elseif idx >= 967
        num_EMG = num_EMG + 1;
    end
end

noisy_inputs_EOG    = zeros(num_EOG, 800);
noisy_outputs_EOG    = zeros(num_EOG, 800);
ground_truth_EOG    = zeros(num_EOG, 800);
noisy_inputs_Motion = zeros(num_Motion, 800);
noisy_outputs_Motion = zeros(num_Motion, 800);
ground_truth_Motion = zeros(num_Motion, 800);
noisy_inputs_EMG    = zeros(num_EMG, 800);
noisy_outputs_EMG    = zeros(num_EMG, 800);
ground_truth_EMG    = zeros(num_EMG, 800);

count_EOG = 1;
count_Motion = 1;
count_EMG = 1;

for i = 1:length(noisy_detect)
    idx = noisy_detect(i);
    if idx < 345
        noisy_inputs_EOG(count_EOG, :)  = z_test_noisy(idx, :);
        noisy_outputs_EOG(count_EOG, :) = z_decoded_layer(idx, :);
        ground_truth_EOG(count_EOG, :)  = z_test_clean(idx, :);
        count_EOG = count_EOG + 1;
    elseif idx >= 345 && idx < 967
        noisy_inputs_Motion(count_Motion, :)  = z_test_noisy(idx, :);
        noisy_outputs_Motion(count_Motion, :) = z_decoded_layer(idx, :);
        ground_truth_Motion(count_Motion, :)  = z_test_clean(idx, :);
        count_Motion = count_Motion + 1;
    elseif idx >= 967
        noisy_inputs_EMG(count_EMG, :)  = z_test_noisy(idx, :);
        noisy_outputs_EMG(count_EMG, :) = z_decoded_layer(idx, :);
        ground_truth_EMG(count_EMG, :)  = z_test_clean(idx, :);
        count_EMG = count_EMG + 1;
    end
end


timeRRMSE_clean = zeros(482,800);
timeRMSE_clean = zeros(482,800);
freqRRMSE_clean = zeros(482,800);
freqRMSE_clean = zeros(482,800);
CC_clean = zeros(482,800);

timeRRMSE_EOG = zeros(482,800);
timeRMSE_EOG = zeros(482,800);
freqRRMSE_EOG = zeros(482,800);
freqRMSE_EOG = zeros(482,800);
CC_EOG = zeros(482,800);

timeRRMSE_Motion = zeros(482,800);
timeRMSE_Motion = zeros(482,800);
freqRRMSE_Motion = zeros(482,800);
freqRMSE_Motion = zeros(482,800);
CC_Motion = zeros(482,800);

timeRRMSE_EMG = zeros(482,800);
timeRMSE_EMG = zeros(482,800);
freqRRMSE_EMG = zeros(482,800);
freqRMSE_EMG = zeros(482,800);
CC_EMG = zeros(482,800);

[timeRRMSE_clean, timeRMSE_clean, freqRRMSE_clean, freqRMSE_clean, CC_clean] = evaluate_metrics(clean_inputs, clean_outputs);
[timeRRMSE_EOG, timeRMSE_EOG, freqRRMSE_EOG, freqRMSE_EOG, CC_EOG] = evaluate_metrics(ground_truth_EOG, noisy_outputs_EOG);
[timeRRMSE_Motion, timeRMSE_Motion, freqRRMSE_Motion, freqRMSE_Motion, CC_Motion] = evaluate_metrics(ground_truth_Motion, noisy_outputs_Motion);
[timeRRMSE_EMG, timeRMSE_EMG, freqRRMSE_EMG, freqRMSE_EMG, CC_EMG] = evaluate_metrics(ground_truth_EMG, noisy_outputs_EMG);

fprintf('\n EEG clean input results:\n');
fprintf('RRMSE-Time: mean = %.4f , std = %.4f\n', mean(timeRRMSE_clean), std(timeRRMSE_clean));
fprintf('RRMSE-Freq: mean = %.4f , std = %.4f\n', mean(freqRRMSE_clean), std(freqRRMSE_clean));
fprintf('CC:         mean = %.4f , std = %.4f\n', mean(CC_clean), std(CC_clean));

fprintf('\n EEG EOG input results:\n');
fprintf('RRMSE-Time: mean = %.4f , std = %.4f\n', mean(timeRRMSE_EOG), std(timeRRMSE_EOG));
fprintf('RRMSE-Freq: mean = %.4f , std = %.4f\n', mean(freqRRMSE_EOG), std(freqRRMSE_EOG));
fprintf('CC:         mean = %.4f , std = %.4f\n', mean(CC_EOG), std(CC_EOG));

fprintf('\n EEG Motion input results:\n');
fprintf('RRMSE-Time: mean = %.4f , std = %.4f\n', mean(timeRRMSE_Motion), std(timeRRMSE_EOG));
fprintf('RRMSE-Freq: mean = %.4f , std = %.4f\n', mean(freqRRMSE_Motion), std(freqRRMSE_Motion));
fprintf('CC:         mean = %.4f , std = %.4f\n', mean(CC_Motion), std(CC_Motion));

fprintf('\n EEG EMG input results:\n');
fprintf('RRMSE-Time: mean = %.4f , std = %.4f\n', mean(timeRRMSE_EMG), std(timeRRMSE_EMG));
fprintf('RRMSE-Freq: mean = %.4f , std = %.4f\n', mean(freqRRMSE_EMG), std(freqRRMSE_EMG));
fprintf('CC:         mean = %.4f , std = %.4f\n', mean(CC_EMG), std(CC_EMG));












function [timeRRMSE, timeRMSE, freqRRMSE, freqRMSE, CC] = evaluate_metrics(true_signals, pred_signals)
% EVALUATE_METRICS 计算信号重建的评估指标
%
%   输入:
%       true_signals - 真值信号矩阵，每行一个信号
%       pred_signals - 预测（重建）信号矩阵，每行一个信号
%
%   输出:
%       timeRRMSE  - 时间域相对均方根误差 (RRMSE)，每个信号一个值
%       timeRMSE   - 时间域均方根误差 (RMSE)
%       freqRRMSE  - 频域相对均方根误差（基于PSD计算）
%       freqRMSE   - 频域均方根误差（基于PSD计算）
%       CC         - 每个信号的皮尔逊相关系数

nSignals = size(true_signals, 1);

% 预分配输出变量（列向量，每行对应一个信号）
timeRRMSE = zeros(nSignals, 1);
timeRMSE  = zeros(nSignals, 1);
freqRRMSE = zeros(nSignals, 1);
freqRMSE  = zeros(nSignals, 1);
CC        = zeros(nSignals, 1);

% 设置频域参数，参考 Python 中的 nperseg=200, nfft=800, fs=200, 默认noverlap=nperseg/2
nperseg = 200;
nfft = 800;
fs = 200;
noverlap = nperseg / 2;
window = hann(nperseg);

for i = 1:nSignals
    % 获取第 i 个信号（假设信号为行向量）
    x_true = true_signals(i, :);
    x_pred = pred_signals(i, :);
    
    %% 时间域指标计算
    % RMSE: sqrt(mean((true - pred).^2))
    rmse_val = sqrt(mean((x_true - x_pred).^2));
    timeRMSE(i) = rmse_val;
    % RRMSE: RMSE / RMS(true)
    rms_true = rms(x_true);
    timeRRMSE(i) = rmse_val / rms_true;
    
    %% 频域指标计算
    % 利用 pwelch 计算功率谱密度（PSD）
    [pxx_true, ~] = pwelch(x_true, window, noverlap, nfft, fs);
    [pxx_pred, ~] = pwelch(x_pred, window, noverlap, nfft, fs);
    % 频域 RMSE
    rmse_psd = sqrt(mean((pxx_true - pxx_pred).^2));
    freqRMSE(i) = rmse_psd;
    % 频域 RRMSE: RMSE(PSD) / RMS(PSD(true))
    rms_psd_true = rms(pxx_true);
    freqRRMSE(i) = rmse_psd / rms_psd_true;
    
    %% 计算皮尔逊相关系数（CC）
    % MATLAB 的 corr 函数需要列向量，因此转置
    CC(i) = corr(x_true', x_pred');
end

end