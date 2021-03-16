datapath='E:\projects b16\pranay\400X\';

% Image Datastore
imds=imageDatastore(datapath, ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.8,0.1,0.1,'randomized');
net=shufflenet;

%analyzeNetwork(net)

net.Layers(1)
inputSize = net.Layers(1).InputSize;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 


[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
%plot(lgraph)
%ylim([0,20])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:25) = freezeWeights(layers(1:25));
lgraph = createLgraphUsingConnections(layers,connections);


pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


miniBatchSize=32;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options=trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.002, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
idx = randperm(numel(imdsValidation.Files),8);
figure
for i = 1:6
    subplot(3,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
[YPred1,probs1] = classify(net,augimdsTest);
accuracy1 = mean(YPred1 == imdsTest.Labels)
idx = randperm(numel(imdsTest.Files),8);
figure
for i = 1:6
    subplot(3,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred1(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
[predicted_labels,posterior]=classify(net,augimdsTest);



% Actual Labels
actual_labels=imdsTest.Labels;

% Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix: shufflenet');

S=confusionmatStats(actual_labels,predicted_labels)

%[c,cm] = confusion(actual_labels,predicted_labels)
%fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
%fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
%plotroc(actual_labels,predicted_labels)
%{
ACTUAL=actual_labels
PREDICTED=predicted_labels
EVAL = Evaluate(ACTUAL,PREDICTED) 
%}
%stats = confusionmatStats(actual_labels,predicted_labels)

%plotroc(actual_labels,predicted_labels)