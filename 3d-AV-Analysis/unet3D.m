function [predictedLabels, groundTruthLabels, diceResult, net, vol] ...
    = unet3D(trainingData, validationData,testingData, inputSize, options)
%   Here is the U-Net for 3D-Volumes located
%   There are 4 parts: The Network architekture, the Training Part, the
%   Prediction Part and a part to calculate the Dice Result. A lot of the
%   Training Parameters are defined at the beginning of the script_3dunet.
%   At the End of the Training Part the Model to just Predict is specified.
%   You can download this Model to directly predict with the Supplementary
%   Material. 


nPixels                = sum(options.tbl.PixelCount);
frequency              = options.tbl.PixelCount / nPixels;
invFrequency           = 1./frequency;
weightFactors          = [options.invffac(1)*invFrequency(1); ...
                          options.invffac(2)*invFrequency(2)];
str                    = sprintf(...
                         'InvFreq-> Background : %5.2f Vessels : %5.2f',...
                         invFrequency(1), invFrequency(2));
fprintf([str '\n'])
inputPatchSize         = inputSize;

%% Network architecture
[lgraph, ~]            = unet3dLayers(inputPatchSize, ...
                         options.numClasses, 'ConvolutionPadding', ...
                         'same', 'EncoderDepth', options.encoderDepth, ...
                         'NumFirstEncoderFilters', options.numFilters);
outputLayer            = dicePixelClassificationLayer('Name','Output');
lgraph                 = replaceLayer(lgraph,'Segmentation-Layer', ...
                         outputLayer);
inputLayer             = image3dInputLayer(inputPatchSize, ...
                         'Normalization', options.normtype, 'Name', ...
                         'ImageInputLayer');
outputLayer            = pixelClassificationLayer('Classes', ...
                         options.tbl.Name, 'ClassWeights', ...
                         weightFactors, 'Name', 'Output');
lgraph                 = replaceLayer(lgraph,'ImageInputLayer',inputLayer);
lgraph                 = replaceLayer(lgraph,'Output',outputLayer);

if options.modelrepitition == 1
    analyzeNetwork           (lgraph)
end


%% Train network
opts                   = trainingOptions(     'adam', 'MaxEpochs', ...
                                                options.maxEpochs, ...
                                          'InitialLearnRate',5e-7, ...
                                  'LearnRateSchedule','piecewise', ...
                                          'LearnRateDropPeriod',5, ...
                                       'LearnRateDropFactor',0.95, ...
                                  'ValidationData',validationData, ...
                                        'ValidationFrequency',100, ...
                                      'Plots','training-progress', ...
                                                   'Verbose',true, ...
                            'MiniBatchSize',options.miniBatchSize, ...
                          'CheckPointPath', options.checkpointdir, ...
                                         'Shuffle', 'every-epoch', ...
                                   'L2Regularization', options.L2reg);


if options.doTraining == true
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    if options.resumeTraining == true
        cd(options.checkpointdir)
        tmp     = dir('*.mat');
        load(tmp(end).name, 'net');
        lgraph  = layerGraph(net);
        cd(datadir)
    end
        
    [net,~]   =   trainNetwork(trainingData,lgraph,opts);
    save             (['trained3DUNet-' modelDateTime '-Epoch-' ...
                     num2str(opts.MaxEpochs) '.mat'],'net');
    outPatchSize   = inputSize;
else
    inputPatchSize = inputSize;
    outPatchSize   = inputSize;
    load('trained3DUNet-13-Feb-2020-11-35-56-Epoch-100.mat'); %#ok<LOAD>
end

%% Predict
id = 1;
while hasdata(testingData.voldsTest)
    disp(['Processing test volume ' num2str(id)]);
    
    tempGroundTruth = read(testingData.pxdsTest);
    groundTruthLabels{id} = tempGroundTruth{1}; %#ok<AGROW>
    vol{id} = read(testingData.voldsTest); %#ok<AGROW>
    
    % Use reflection padding for the test image. 
    % Avoid padding of different modalities.
    volSize                            = size(vol{id},(1:3));
    padSizePre                         = (inputPatchSize(1:3) - ...
                                         outPatchSize(1:3))/2;
    padSizePost                        = (inputPatchSize(1:3) - ...
                                         outPatchSize(1:3))/2 + ...
                                         (outPatchSize(1:3) - ...
                                         mod(volSize,outPatchSize(1:3)));
    volPaddedPre                       = padarray(vol{id},padSizePre, ...
                                         'symmetric','pre');
    volPadded                          = padarray(volPaddedPre, ...
                                         padSizePost,'symmetric','post');
    [heightPad, widthPad, depthPad, ~] = size(volPadded);
    [height, width, depth, ~]          = size(vol{id});
    
    tempSeg                            = categorical(zeros([height, ...
                                         width, depth], 'uint8'), ...
                                         [0 1], options.classNames);
    
    % Overlap-tile strategy for segmentation of volumes.
    for k = 1:outPatchSize(3):depthPad-inputPatchSize(3)+1
        for j = 1:outPatchSize(2):widthPad-inputPatchSize(2)+1
            for i = 1:outPatchSize(1):heightPad-inputPatchSize(1)+1
                patch = volPadded( i:i+inputPatchSize(1)-1,...
                    j:j+inputPatchSize(2)-1,...
                    k:k+inputPatchSize(3)-1,:);
                patchSeg = semanticseg(patch,net);
                tempSeg(i:i+outPatchSize(1)-1, ...
                    j:j+outPatchSize(2)-1, ...
                    k:k+outPatchSize(3)-1) = patchSeg;
            end
        end
    end
    
    % Crop out the extra padded region.
    tempSeg             = tempSeg(1:height,1:width,1:depth);
    % Save the predicted volume result.
    predictedLabels{id} = tempSeg; %#ok<AGROW>
    id                  = id + 1;
end



%% Calculate dice result
diceResult                = zeros(length(testingData.voldsTest.Files),2);

for j = 1:length(vol)
    diceResult(j,:)       = dice(groundTruthLabels{j},predictedLabels{j});
end
