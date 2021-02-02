function [trainingData, validationData, testingData, options] = ...
    prepareData(imageDir, inputSize, options)
%prepareData this Function does the Preparation of the Data for the U-Net
% It mainly creates imageDatastore and pixelLabelDatastore with the raw
% Images and labeled Images. Then it crops small patches of a size
% specified by input size. On these little Pieces it starts training.
volReader      = @ (x) matRead(x);
labeledReader  = @ (x) matRead(x)>0;
volLoc         = fullfile(imageDir,'imagesTr');
volds          = imageDatastore(volLoc, ...
                 'FileExtensions','.mat','ReadFcn',volReader);
lblLoc         = fullfile(imageDir,'labelsTr');
pxds           = pixelLabelDatastore(lblLoc,options.classNames, ...
                 options.pixelLabelID, ...
                'FileExtensions','.mat','ReadFcn',labeledReader);
pximds         = pixelLabelImageDatastore(volds,pxds);
tbl            = countEachLabel(pximds);
options.tbl    = tbl;


% calculate min and max intensities for training data
fprintf('TRAINING DATA: \n')
II = zeros(length(volds.Files),2);
for k = 1 : length(volds.Files)
    tmp     = load(volds.Files{k});
    II(k,:) = [min(tmp.tmp(:)) max(tmp.tmp(:))];
    ind1    = strfind(volds.Files{k}, '\');
    ind2    = strfind(volds.Files{k}, '.mat');
    name    = volds.Files{k}(ind1(end)+1:ind2-1);
    disp  ([ 'Image-' num2str(k) ' (' name ')' ' Dim: [' ...
        num2str(size(tmp.tmp)) '] , Intensity: [' num2str(II(k,:)) ']'])

          if size(tmp.tmp,1)< inputSize(1) || size(tmp.tmp,2)< ...
                  inputSize(2) || size(tmp.tmp,3)< inputSize(3) 
              error('Error. \n image size must be => patch size')
          end
end

patchSize             = inputSize;
patchds               = randomPatchExtractionDatastore(volds,pxds, ...
                        patchSize,'PatchesPerImage',options.patchPerImage);
patchds.MiniBatchSize = options.miniBatchSize;
volLocVal             = fullfile(imageDir,'imagesVal');
voldsVal              = imageDatastore(volLocVal, ...
                         'FileExtensions','.mat','ReadFcn',volReader);
lblLocVal             = fullfile(imageDir,'labelsVal');
pxdsVal               = pixelLabelDatastore(lblLocVal,...
                        options.classNames, options.pixelLabelID, ...
                        'FileExtensions','.mat', 'ReadFcn',labeledReader);
dsVal                 = randomPatchExtractionDatastore(voldsVal,...
                        pxdsVal,patchSize,'PatchesPerImage', ...
                        options.patchPerImage);
dsVal.MiniBatchSize   = options.miniBatchSize;
dataSource            = 'Training';
dsTrain               = transform(patchds,@(patchIn)...
                        augmentAndCrop3dPatchVessel(patchIn,dataSource));
dataSource            = 'Validation';
dsVal                 = transform(dsVal,@(patchIn) ...
                        augmentAndCrop3dPatchVessel(patchIn,dataSource));
% calculate min and max intensities for validation data
fprintf('\nVALIDATION DATA: \n')
II = zeros(length(voldsVal.Files),2);
for k = 1 : length(voldsVal.Files)
    tmp     = load(voldsVal.Files{k});
    II(k,:) = [min(tmp.tmp(:)) max(tmp.tmp(:))];
    ind1    = strfind(voldsVal.Files{k}, '\');
    ind2    = strfind(voldsVal.Files{k}, '.mat');
    name    = voldsVal.Files{k}(ind1(end)+1:ind2-1);
    disp  ([ 'Image-' num2str(k) ' (' name ')' ' Dim: [' ...
        num2str(size(tmp.tmp)) '] , Intensity: [' num2str(II(k,:)) ']'])
    if size(tmp.tmp,1)< inputSize(1) || size(tmp.tmp,2)< inputSize(2)|| ...
            size(tmp.tmp,3)< inputSize(3)
        error('Error. \n image size must be => patch size')
    end
end

%Original Code with cropping window
volLocTest   = fullfile(imageDir,'imagesTest');
lblLocTest   = fullfile(imageDir,'labelsTest');
volReader    = @(x) matRead(x);
voldsTest    = imageDatastore(volLocTest, ...
    'FileExtensions', '.mat', 'ReadFcn', volReader);
pxdsTest     = pixelLabelDatastore(lblLocTest, options.classNames, ...
              options.pixelLabelID, ...
              'FileExtensions', '.mat', 'ReadFcn', labeledReader);
          
% Testing code
% calculate min and max intensities for testing data
fprintf('\nTESTING DATA: \n')
II = zeros(length(voldsTest.Files),2);
for k = 1 : length(voldsTest.Files)
    tmp     = load(voldsTest.Files{k});
    II(k,:) = [min(tmp.tmp(:)) max(tmp.tmp(:))];
    ind1    = strfind(voldsTest.Files{k}, '\');
    ind2    = strfind(voldsTest.Files{k}, '.mat');
    name    = voldsTest.Files{k}(ind1(end)+1:ind2-1);
    disp  ([ 'Image-' num2str(k) ' (' name ')' ' Dim: [' ...
        num2str(size(tmp.tmp)) '] , Intensity: [' num2str(II(k,:)) ']'])
    if size(tmp.tmp,1)< inputSize(1) || size(tmp.tmp,2)< inputSize(2)|| ...
            size(tmp.tmp,3)< inputSize(3)
        error('Error. Image size must be => patch size')
    end
end


trainingData          = dsTrain;
validationData        = dsVal; 
testingData.voldsTest = voldsTest;
testingData.pxdsTest  = pxdsTest;
