%input weights data
data_weights = load('model_weights.mat');

% autoencoder layers
layers = [
    imageInputLayer([800,1,1], 'Name', 'input') % input layers (800, 1, 1)

    % encoder
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Stride', [1,1], 'Name', 'enc_conv1')
    reluLayer('Name', 'enc_relu1')
    convolution2dLayer([3,1], 32, 'Padding', 'same', 'Stride', [1,1], 'Name', 'enc_conv2')
    reluLayer('Name', 'enc_relu2')
    convolution2dLayer([3,1], 16, 'Padding', 'same', 'Stride', [1,1], 'Name', 'enc_conv3')
    reluLayer('Name', 'enc_relu3')
    convolution2dLayer([3,1], 4, 'Padding', 'same', 'Stride', [1,1], 'Name', 'enc_conv4')
    reluLayer('Name', 'enc_relu4')

    % decoder
    convolution2dLayer([3,1], 16, 'Padding', 'same', 'Stride', [1,1], 'Name', 'dec_conv1')
    reluLayer('Name', 'dec_relu1')
    convolution2dLayer([3,1], 32, 'Padding', 'same', 'Stride', [1,1], 'Name', 'dec_conv2')
    reluLayer('Name', 'dec_relu2')
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Stride', [1,1], 'Name', 'dec_conv3')
    reluLayer('Name', 'dec_relu3')
    convolution2dLayer([3,1], 1, 'Padding', 'same', 'Stride', [1,1], 'Name', 'dec_output')
    sigmoidLayer('Name', 'sigmoid_output')
];

% input layers bias and weight

weights2D1 = reshape(data_weights.sequential_param_0, [3, 1, 1, 64]);
weights2D2 = reshape(data_weights.sequential_param_2, [3, 1, 64, 32]);
weights2D3 = reshape(data_weights.sequential_param_4, [3, 1, 32, 16]);
weights2D4 = reshape(data_weights.sequential_param_6, [3, 1, 16, 4]);
weights2D5 = reshape(data_weights.sequential_1_param_0, [3, 1, 4, 16]);
weights2D6 = reshape(data_weights.sequential_1_param_2, [3, 1, 16, 32]);
weights2D7 = reshape(data_weights.sequential_1_param_4, [3, 1, 32, 64]);
weights2D8 = reshape(data_weights.sequential_1_param_6, [3, 1, 64]);
bias2D1 = reshape(data_weights.sequential_param_1, [1, 1, 64]);
bias2D2 = reshape(data_weights.sequential_param_3, [1, 1, 32]);
bias2D3 = reshape(data_weights.sequential_param_5, [1, 1, 16]);
bias2D4 = reshape(data_weights.sequential_param_7, [1, 1, 4]);
bias2D5 = reshape(data_weights.sequential_1_param_1, [1, 1, 16]);
bias2D6 = reshape(data_weights.sequential_1_param_3, [1, 1, 32]);
bias2D7 = reshape(data_weights.sequential_1_param_5, [1, 1, 64]);
%bias2D8 = reshape(data_weights.sequential_1_param_7, [1, 1, 64]);

layers(2).Weights = single(weights2D1);
layers(2).Bias = single(bias2D1);
layers(4).Weights = single(weights2D2);
layers(4).Bias = single(bias2D2);
layers(6).Weights = single(weights2D3);
layers(6).Bias = single(bias2D3);
layers(8).Weights = single(weights2D4);
layers(8).Bias = single(bias2D4);
layers(10).Weights = single(weights2D5);
layers(10).Bias = single(bias2D5);
layers(12).Weights = single(weights2D6);
layers(12).Bias = single(bias2D6);
layers(14).Weights = single(weights2D7);
layers(14).Bias = single(bias2D7);
layers(16).Weights = single(weights2D8);
layers(16).Bias = single(data_weights.sequential_1_param_7);

autoencoder = dlnetwork(layers);
save('autoencoder2DNetwork.mat', 'autoencoder');

% show network
analyzeNetwork(autoencoder);
