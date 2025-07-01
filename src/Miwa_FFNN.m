%% 研究室PC
% Sorce Folder
sf = 'C:\Users\ryoya\MasterThesis\MT_Furuie\src';
% Data Folder
df = 'C:\Users\ryoya\MasterThesis\MT_Furuie\data';
% Result Folder
rf = 'C:\Users\ryoya\MasterThesis\MT_Furuie\results\Miwa_FFNN';

%% LapTop
% % Data Folder
% df = 'C:\Users\RYOYA\Documents\修士研究\美和ダムデータ';
% % Result Folder
% rf = 'C:\Users\RYOYA\Documents\修士研究\美和ダムデータ';

%% ========== 設定 ==========

% Model 1: r(t-3), r(t-5), r24(t-1) [7 9 33]

cd(df)
filename = 'Miwa_flood_for_FFNN.xlsx';
inputCols  = [7 9 33];  % 入力列
outputCol = 3;        % 出力列


%% ========= データ読み込み =========
trainData = readtable(filename, 'Sheet', 'Train');
valData   = readtable(filename, 'Sheet', 'Val');
testData  = readtable(filename, 'Sheet', 'Test');

XTrain_raw = trainData{:, inputCols};
YTrain_raw = trainData{:, outputCol};

XVal_raw = valData{:, inputCols};
YVal_raw = valData{:, outputCol};

XTest_raw = testData{:, inputCols};
YTest_raw = testData{:, outputCol};

%% ========= 正規化（z-score） =========
muX = mean(XTrain_raw);
sigmaX = std(XTrain_raw);
XTrain = (XTrain_raw - muX) ./ sigmaX;
XVal   = (XVal_raw   - muX) ./ sigmaX;
XTest  = (XTest_raw  - muX) ./ sigmaX;

muY = mean(YTrain_raw);
sigmaY = std(YTrain_raw);
YTrain = (YTrain_raw - muY) / sigmaY;
YVal   = (YVal_raw   - muY) / sigmaY;
YTest  = (YTest_raw  - muY) / sigmaY;

%% ========= レイヤー定義（シグモイド＋ドロップアウト） =========

numInputs = numel(inputCols);
layers = [
    featureInputLayer(numInputs, "Name", "input")

    fullyConnectedLayer(20, "Name", "fc1")
    sigmoidLayer("Name", "sigmoid1")

    dropoutLayer(0.1, "Name", "dropout1")

    fullyConnectedLayer(10, "Name", "fc2")
    sigmoidLayer("Name", "sigmoid2")

    fullyConnectedLayer(1, "Name", "fc_out")
    regressionLayer("Name", "output")
];

%% ========= 学習オプション（ミニバッチ含む） =========
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 24, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 26, ... % 1エポックで2回検証される設定
    'ValidationPatience', 5, ...  % 5回連続で改善しなければ停止
    'Plots', 'training-progress', ...
    'Verbose', true);

%% ========= 学習実行 =========
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

%% ========= テストデータで予測（正規化戻す） =========
YPred_norm = predict(net, XTest);
YPred = YPred_norm * sigmaY + muY;   % 出力のスケールを元に戻す

%% ========= 評価 =========
YTrue = YTest_raw;  % スケール戻した実測値

% RMSE（平均二乗根誤差）
rmse = sqrt(mean((YPred - YTrue).^2));

% NSE（Nash-Sutcliffe Efficiency）
numerator = sum((YTrue - YPred).^2);
denominator = sum((YTrue - mean(YTrue)).^2);
nse = 1 - numerator / denominator;

% R2（二乗決定係数）
SS_res = sum((YTrue - YPred).^2);
SS_tot = sum((YTrue - mean(YTrue)).^2);
R2 = 1 - SS_res / SS_tot;

% 表示
fprintf('Test RMSE: %.3f\n', rmse);
fprintf('Test NSE:  %.3f\n', nse);
fprintf('Test R^2:   %.3f\n', R2);

%% ========= 結果保存 =========
cd(rf)

save('Miwa_FFNN_trained_1_6.mat', 'net', 'muX', 'sigmaX', 'muY', 'sigmaY');

close all
figure;
plot(YPred, 'r-o', 'DisplayName', '予測濁度');
hold on;
plot(YTrue, 'b-*', 'DisplayName', '実測濁度');
legend; grid on;
xlabel('Sample'); ylabel('濁度'); title('テストデータ予測結果');
saveas(gcf, 'PredictionResult_1_6.png');

% グラフ作成
close all
figure;
plot(info.TrainingLoss, '-o', 'DisplayName', 'Training Loss');
hold on;
if isfield(info, 'ValidationLoss')
    plot(info.ValidationLoss, '-*', 'DisplayName', 'Validation Loss');
end
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');
legend;
grid on;

% 保存
saveas(gcf, 'LossHistory_1_6.png');