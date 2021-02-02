clearvars
close all
clc
imageDir = uigetdir(pwd,'Select the main data directory');

options.doTraining      = false;% Shall we train or just predict?
                                % to just predict set false
inputSize               = [64 64 64]; % Set the patch size
                                      % Watch, that imageSize> patchSize
options.numClasses      = 2;          
options.classNames      = ["background", "vessels"];
options.pixelLabelID    = [0 1];
options.patchPerImage   = 36;   % How many patches of the size specified
                                %(64*64*64 pixels) should be automatically cropped?
options.miniBatchSize   = 4;    % On how many patches should be worked at
                                % the same time?
options.maxEpochs       = 100;  % Set the amount of training epochs here
options.encoderDepth    = 3;    % How deep shall the U-Net be?
options.numFilters      = 32;   % How many Filters should be applied?
options.testVis         = 1;    % set to 1, if you want to have gifs and
                                % video of Segmentation
options.resumeTraining  = false;
options.checkpointdir   = [imageDir '\trained3DUNet'];
options.invffac         = [1  1];
options.normtype        = 'rescale-zero-one'; % Scale between 0 and 1
%'zerocenter'| 'zscore' | 'rescale-symmetric' | 'rescale-zero-one' | 'none'
% Other Normalisation Options
options.L2reg           = 0.001; % see Explanation in Paper
totalitr                = 1;
options.modelrepitition = 1; % dont change this
modelDateTime           = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
v                       = VideoWriter(['vid_' modelDateTime '.avi']);
for modelrepitition = 1 : totalitr
    options.modelrepitition = modelrepitition;
    [trainingData, validationData, testingData, options] = prepareData(...
        imageDir, inputSize, options);
    
    [predictedLabels, groundTruthLabels, diceResult, ...
        net, vol]                          = unet3D( ...
        trainingData, ...
        validationData, ...
        testingData, ...
        inputSize, options);
    
    if  options.testVis == 1
        for p = 1 : numel(testingData.voldsTest.Files)
            ind1           = strfind(testingData.voldsTest.Files{p}, '\');
            ind2           = strfind(testingData.voldsTest.Files{p}, ...
                '.mat');
            name           = testingData.voldsTest. ...
                Files{p}(ind1(end)+1:ind2-1);
            filename1      = ['labeledVolume_' name '.gif'];
            filename2      = ['predictedVolume_' name '.gif'];
            CameraUpVector = [-0.2 -1 -0.2];
            CameraPosition = [-2 -1 5];
            vec            = linspace(CameraPosition(1),3,100)';
            xcam           = vec;
            ycam           = CameraPosition(2)*ones(size(vec,1),1);
            zcam           = CameraPosition(3)*ones(size(vec,1),1);
            myPosition     = [xcam ycam zcam];
            vol3d          = vol{p};
            
            % 3D volume display for ground truth data
            viewPnlTruth   = uipanel(figure,'Title', ...
                'Ground-Truth Labeled Volume');
            hTruth         = labelvolshow(groundTruthLabels{p}, ...
                vol3d, 'Parent', viewPnlTruth, ...
                'LabelColor', [0 0 0;1 0 0], ...
                'VolumeThreshold',0.12, 'LabelOpacity', ...
                [0.01; 0.1], 'BackgroundColor', [0 0.9 0]);
            hTruth.          LabelVisibility(1) = 0;
            hTruth.          LabelVisibility(2) = 0;
            hTruth.          CameraUpVector     = CameraUpVector;
            hTruth.          CameraPosition     = CameraPosition;
            
            % Video capturing/saving for ground truth data
            for idx = 1:length(vec)
                % Update current view.
                hTruth.CameraPosition = myPosition(idx,:);
                % Use getframe to capture image.
                I = getframe(gcf);
                [indI,cm] = rgb2ind(I.cdata,256);
                % Write frame to the GIF File.
                if idx == 1
                    imwrite(indI, cm, filename1, 'gif', ...
                        'Loopcount', inf, 'DelayTime', 0.01);
                else
                    imwrite(indI, cm, filename1, 'gif', 'WriteMode', ...
                        'append', 'DelayTime', 0.01);
                end
            end
            
            
            % 3D volume display for predicted data
            viewPnlPred    = uipanel(figure,'Title', ...
                'Predicted Labeled Volume');
            hPred          = labelvolshow(predictedLabels{p},vol3d, ...
                'Parent', viewPnlPred, 'LabelColor', ...
                [0 0 0;1 0 0], 'VolumeThreshold',0.18, ...
                'LabelOpacity', [0.01; 0.1], ...
                'BackgroundColor', [0 0.9 0]);
            hPred.LabelVisibility(1) = 0;
            hPred.CameraUpVector     = CameraUpVector;
            hPred.CameraPosition     = CameraPosition;
            
            % Video capturing/saving for predicted data
            for idx = 1:length(vec)
                hPred.CameraPosition = myPosition(idx,:);
                I                    = getframe(gcf);
                [indI,cm]            = rgb2ind(I.cdata,256);
                if idx == 1
                    imwrite(indI, cm, filename2, 'gif', ...
                        'Loopcount', inf, 'DelayTime', 0.01);
                else
                    imwrite(indI, cm, filename2, 'gif', 'WriteMode', ...
                        'append', 'DelayTime', 0.01);
                end
            end
            
            hFig = figure;
            title('Labeled  (Left) vs. Network Prediction (Right)')
            open(v)
            for zID = 1 : size(vol3d,3)
                zSliceGT   = labeloverlay(uint8(vol3d(:,:,zID)), ...
                    groundTruthLabels{p}(:,:,zID));
                zSlicePred = labeloverlay(uint8(vol3d(:,:,zID)), ...
                    predictedLabels{p}(:,:,zID));
                montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5)
                title(['Slicenumber : ' num2str(zID) ' / ' ...
                    num2str(size(vol3d,3))])
                pause(0.01)
                frame = getframe(gcf);
                writeVideo(v,frame);
            end
            close(v)
        end
        % Calculate and display dice results
        diceResult     = zeros(length(testingData.voldsTest.Files),2);
        
        for j = 1:length(vol)
            diceResult(j,:)= dice(groundTruthLabels{j},predictedLabels{j});
        end
        
        meanDiceBackground = mean(diceResult(:,1));
        meanDiceVessel     = mean(diceResult(:,2));
        disp               (['Average Dice score of background across ',...
            num2str(j), ' test volumes = ', ...
            num2str(meanDiceBackground)])
        disp               (['Average Dice score of vessels across ',...
            num2str(j), ' test volumes = ', ...
            num2str(meanDiceVessel)])
    end
    
    % Create box plot
    createBoxplot      = true;
    
    if createBoxplot
        figure
        boxplot(diceResult)
        diceArray{modelrepitition} = diceResult; %#ok<SAGROW>
        title('Test Set Dice Accuracy')
        xticklabels(options.classNames)
        ylabel('Dice Coefficient')
    end
    
end

% Plot all dice Results
figure;
for i = 1 : numel(diceArray)
    plot(i,mean(diceArray{i}(:,1)), '-*r')
    hold on
    plot(i,mean(diceArray{i}(:,2)), '-ob')
end
grid on
cd ([imageDir '/prediction'])
save('predictedLabels','predictedLabels', '-v7.3');
cd (imageDir)
save diceArray diceArray