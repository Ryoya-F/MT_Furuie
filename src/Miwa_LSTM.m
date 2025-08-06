%% 0. 環境設定とクリア
clear; clc; close all;

% Sorce Folder
sf = 'C:\Users\ryoya\MasterThesis\MT_Furuie\src';
% Data Folder
df = 'C:\Users\ryoya\MasterThesis\MT_Furuie\data\Miwa_LSTM_Data\Trial_1';
% Result Folder
rf = 'C:\Users\ryoya\MasterThesis\MT_Furuie\results\Miwa_LSTM\Trial_1';

%% 1. データ読み込み (フォルダからの自動読み込みとシーケンス生成)

disp('データを読み込んでいます...');
cd(df);

% データフォルダのパスを指定
trainDataFolder = 'train_data/';
validationDataFolder = 'val_data/';
testDataFolder = 'test_data/';

% 入力シーケンスの長さ (240時間)
sequenceLength = 240;

% 入力・出力変数の列数を指定
xIdx = 3; % 流域平均雨量
yIdx = 16; % 濁度

disp('データの前処理を開始します (各イベント内でシーケンス生成)...');

% --- 学習データの処理 ---
% まず、学習データ全体の生データを読み込み、標準化パラメータを計算
disp('学習データの生データを読み込み、標準化パラメータを計算中...');
[XTrainRawCombined, YTrainRawCombined, trainFiles] = readRawDataFromFolder(trainDataFolder, xIdx, yIdx, sequenceLength);
[XTrainNormCombined, XMean, XStd] = zscore(XTrainRawCombined);
[YTrainNormCombined, YMean, YStd] = zscore(YTrainRawCombined);
disp('標準化パラメータ計算完了。');

%% 2. データシーケンスの作成
% 各学習イベントファイルごとにシーケンスを生成し、結合

XTrain = {};
YTrain = {};
for i = 1:length(trainFiles)
    filePath = fullfile(trainDataFolder, trainFiles{i});
    tbl = readtable(filePath, NumHeaderLines=1);
    currentFeatures = tbl{:, xIdx};
    currentTargets = tbl{:, yIdx};

    % 取得した標準化パラメータで各イベントデータを標準化
    currentFeaturesNorm = (currentFeatures - XMean) ./ XStd;
    currentTargetsNorm = (currentTargets - YMean) ./ YStd;

    [currentX, currentY] = createSequences(currentFeaturesNorm, currentTargetsNorm, sequenceLength);
    XTrain = [XTrain; currentX];
    YTrain = [YTrain; currentY];
end
disp(['学習シーケンス数: ', num2str(numel(XTrain))]);

YTrain = cell2mat(YTrain);

% --- 検証データの処理 ---
disp('検証データのシーケンスを生成中...');
[~, ~, validationFiles] = readRawDataFromFolder(validationDataFolder, xIdx, yIdx, sequenceLength); % ファイルリストのみ取得
XValidation = {};
YValidation = {};
for i = 1:length(validationFiles)
    filePath = fullfile(validationDataFolder, validationFiles{i});
    tbl = readtable(filePath, NumHeaderLines=1);
    currentFeatures = tbl{:, xIdx};
    currentTargets = tbl{:, yIdx};

    % 学習データで得た標準化パラメータで標準化
    currentFeaturesNorm = (currentFeatures - XMean) ./ XStd;
    currentTargetsNorm = (currentTargets - YMean) ./ YStd;

    [currentX, currentY] = createSequences(currentFeaturesNorm, currentTargetsNorm, sequenceLength);
    XValidation = [XValidation; currentX];
    YValidation = [YValidation; currentY];
end
disp(['検証シーケンス数: ', num2str(numel(XValidation))]);

YValidation = cell2mat(YValidation);

% --- テストデータの処理 ---
disp('テストデータのシーケンスを生成中...');
[~, ~, testFiles] = readRawDataFromFolder(testDataFolder, xIdx, yIdx, sequenceLength); % ファイルリストのみ取得
XTest = {};
YTest = {};
for i = 1:length(testFiles)
    filePath = fullfile(testDataFolder, testFiles{i});
    tbl = readtable(filePath, NumHeaderLines=1);
    currentFeatures = tbl{:, xIdx};
    currentTargets = tbl{:, yIdx};

    % 学習データで得た標準化パラメータで標準化
    currentFeaturesNorm = (currentFeatures - XMean) ./ XStd;
    currentTargetsNorm = (currentTargets - YMean) ./ YStd;

    [currentX, currentY] = createSequences(currentFeaturesNorm, currentTargetsNorm, sequenceLength);
    XTest = [XTest; currentX];
    YTest = [YTest; currentY];
end
disp(['テストシーケンス数: ', num2str(numel(XTest))]);

YTest = cell2mat(YTest);

disp('データの前処理が完了しました。');


%% 3. LSTMネットワークアーキテクチャの定義
disp('LSTMネットワークを構築しています...');
numFeatures = size(XTrain{1}, 2);
numResponses = size(YTrain(1, 1), 2);
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(numResponses)];

disp('ネットワーク構築が完了しました。');

%% 4. 学習オプションの設定
disp('学習オプションを設定しています...');
maxEpochs = 50;
learningRate = 0.001;
miniBatchSize = 24;
validationFrequency = floor(numel(XTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', learningRate, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', 5, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', 1);
disp('学習オプションの設定が完了しました。');

%% 5. ネットワークの学習
disp('ネットワークの学習を開始します...');
net = trainnet(XTrain, YTrain, layers, "mse", options);
disp('ネットワークの学習が完了しました。');

%% 6. テストデータでの評価と結果の表示
disp('テストデータでモデルを評価しています...');
YPredNorm = minibatchpredict(net, XTest);
YPred = YPredNorm .* YStd + YMean;

YTestActual = YTest .* YStd + YMean;

mse = mean((YPred - YTestActual).^2);
rmse = sqrt(mse);
mae = mean(abs(YPred - YTestActual));
R_squared = 1 - sum((YTestActual - YPred).^2) / sum((YTestActual - mean(YTestActual)).^2);


disp('テストデータでの評価結果:');
disp(['  MSE (Mean Squared Error): ', num2str(mse)]);
disp(['  RMSE (Root Mean Squared Error): ', num2str(rmse)]);
disp(['  MAE (Mean Absolute Error): ', num2str(mae)]);
disp(['  R-squared: ', num2str(R_squared)]);
disp(['  NSE:   ', num2str(nse)]);

figure;
plot(YTestActual, 'b', 'LineWidth', 1.5);
hold on;
plot(YPred, 'r--', 'LineWidth', 1.5);
legend('実測値 (Actual Turbidity)', '予測値 (Predicted Turbidity)');
title('テストデータでの濁度予測');
xlabel('時間ステップ');
ylabel('濁度 (ppm)');
grid on;
hold off;
disp('予測結果の可視化が完了しました。');


%% ヘルパー関数: フォルダ内のExcelファイルを読み込み、生データを結合する関数
function [combinedX, combinedY, fileNames] = readRawDataFromFolder(folderPath, xIdx, yIdx, sequenceLength)

    % 入力引数：folderPath : 読み込みたいフォルダのパス（train_data, val_data, test_data）
    %           xIdx : 入力変数に対応する列数（例：[3 7]　3列目は雨量、7列目は流量）
    %           yIdx : 出力変数に対応する列数（例：16　16列目が濁度)

    % 出力引数 : combinedX : フォルダ内の入力変数データを全て結合したもの
    %            combinedY : フォルダ内の出力変数データを全て結合したもの（ただし、readTime分は除く）
    %            fileNames : フォルダ内のExcelファイル名のリスト

    % 指定されたフォルダ内のすべての.xlsxファイルを取得
    files = dir(fullfile(folderPath, '*.xlsx'));
    fileNames = {files.name}; % ファイル名リスト

    % 取り出したデータを格納するセル
    allX = cell(length(fileNames), 1);
    allY = cell(length(fileNames), 1);

    % エラー検知
    if isempty(fileNames)
        warning('フォルダ %s にExcelファイルが見つかりませんでした。', folderPath);
        combinedX = [];
        combinedY = [];
        return;
    end

    for i = 1:length(fileNames)
        filePath = fullfile(folderPath, fileNames{i});
        try
            tbl = readtable(filePath, NumHeaderLines=1);
            allX{i} = tbl{:, xIdx};
            allY{i} = tbl{sequenceLength+1:end, yIdx};
        catch ME
            warning('ファイル %s の読み込み中にエラーが発生しました: %s', filePath, ME.message);
            allX{i} = [];
            allY{i} = [];
        end
    end

    % Cell配列内の全データを結合 (行方向に連結)
    combinedX = vertcat(allX{:});
    combinedY = vertcat(allY{:});
end

%% ヘルパー関数: シーケンス生成 (Slide Window Generator)
function [X, Y] = createSequences(features, targets, sequenceLength)
    numObservations = size(features, 1);
    if numObservations <= sequenceLength
        X = {};
        Y = {};
        % warning('データ長がシーケンス長以下です。シーケンスは生成されません。'); % 各イベントで出るためコメントアウト
        return;
    end

    X = cell(numObservations - sequenceLength, 1);
    Y = cell(numObservations - sequenceLength, 1);

    for i = 1:(numObservations - sequenceLength)
        X{i} = features(i+1 : i + sequenceLength, :);
        Y{i} = targets(i + sequenceLength, :);
    end
end
