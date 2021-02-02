clearvars
close all
clc

Raw_images_dir = uigetdir(pwd, 'Where is the raw data located?');
cd (Raw_images_dir)
files = dir('*.tif');

U_Net_dir = uigetdir(pwd, 'Where should U-Net folders be located?');

%% Important parameters for Normalisation
options.sigma                   = 1.2; % Parameter for Smoothing
options.rangeMin                = 25;  % Minimum, when Normalizing
options.rangeMax                = 100; % Maximum, when Normalizing
%% Set parameters & pre-allocation
parameters.redfac    = 1;
parameters.datatype  = 'raw'; % label, whatever(for cropped images)
parameters.redfacz   = 1;

counter = 0;
hFig = figure;
for k = 1: length(files)
    parameters.filename  = files(k).name;
    numimgs = size(imfinfo(parameters.filename),1);
    %% Read the raw data
    rawfile  = dir(parameters.filename);
    rawData  = imread(rawfile.name);
    I        = zeros(ceil(size(rawData,1)* parameters.redfac), ...
        ceil(size(rawData,2)* parameters.redfac), ...
        ceil(numimgs* parameters.redfacz));
    i                   = 1;
    for j = 1 : parameters.redfacz : numimgs
        
        idx1 = strfind(parameters.filename, 'cropped');
        idx2 = strfind(parameters.filename, 'labeling');
        idx3 = strfind(parameters.filename, '_');
        idx4 = strfind(parameters.filename, '.');
        if ~isempty(idx3) && ~isempty(idx4)
            sno  = parameters.filename(idx3(end)+1:idx4-1);
            if ~isempty(idx2)
                parameters.datatype  = 'label';
            end
            if strcmp(parameters.datatype, 'label') ~= 1 && ...
                    size(imread(rawfile.name, j),3) > 1
                tmpI     = rgb2gray(imread(rawfile.name, j));
            else
                tmpI     = imread(rawfile.name, j);
            end
        else
            tmpI     = imread(rawfile.name, j);
        end
        I(:,:,i) = tmpI(1:ceil(1/parameters.redfac):end, ...
            1:ceil(1/parameters.redfac):end);
        disp      (['Reading slice number : ' num2str(i)])
        i        = i + 1;
    end
    
    [X,Y,Z] = size(I);
    imshow(I(:,:, floor(Z/2)),[]);
    tt = strrep(parameters.filename, '_', '__');
    title(tt)
    impixelinfo;
    hFig.WindowState = 'maximized';
    
    %% Save files
    savename = strcat(parameters.filename(1:idx4-1), '.mat');
    tmp = I(1:X,1:Y,:);
    if files(k).bytes > 2.0e+08
        save(savename, 'tmp', '-v7.3');
    else
        save(savename, 'tmp');
    end
end

%% Create training directories

cd (U_Net_dir);

if ~exist([U_Net_dir '/prediction'],'dir')
    mkdir([U_Net_dir '/prediction']);
end

if ~exist([U_Net_dir '\\trained3DUNet'],'dir')
    mkdir([U_Net_dir '\\trained3DUNet']);
end

if ~exist([U_Net_dir '\imagesTr'],'dir')
    mkdir([U_Net_dir '\imagesTr']);
end

if ~exist([U_Net_dir '\labelsTr'],'dir')
    mkdir([U_Net_dir '\labelsTr']);
end

if ~exist([U_Net_dir '\imagesVal'],'dir')
    mkdir([U_Net_dir '\imagesVal']);
end

if ~exist([U_Net_dir '\labelsVal'],'dir')
    mkdir([U_Net_dir '\labelsVal']);
end

if ~exist([U_Net_dir '\imagesTest'],'dir')
    mkdir([U_Net_dir '\imagesTest']);
end

if ~exist([U_Net_dir '\labelsTest'],'dir')
    mkdir([U_Net_dir '\labelsTest']);
end


%% Apply Normalisation

x= 64; % Minimal dimensions for cropped images
y= 64; 
z= 64; 
cd (Raw_images_dir);

volReader      = @(x) matRead(x);
volds          = imageDatastore(Raw_images_dir, ...
                 'FileExtensions','.mat','ReadFcn', volReader);

II = zeros(length(volds.Files),2);
for k = 1 : length(volds.Files)
    tmp     = load(volds.Files{k});
    disp([ 'Image' num2str(k)  '--size :' num2str(size(tmp.tmp))]);
    II(k,:) = [min(tmp.tmp(:)) max(tmp.tmp(:))];
    disp([ 'IntensityMin : ' num2str(II(k,1))  ', IntensityMax : ' ...
        num2str(II(k,2))]);
    if size(tmp.tmp,1)< x || size(tmp.tmp,2)< y || ...
            size(tmp.tmp,3)< z
        error('Error. Image size must be => patch size')
    end
    
    subplot(2,2,1)
    imshow(tmp.tmp(:,:,floor(size(tmp.tmp,3)/2)),[])
    impixelinfo
    title(volds.Files{k})
    disp(volds.Files{k})
    hFig.WindowState = 'maximized';
    impixelinfo
    im        = tmp.tmp;
    im_smooth = imgaussfilt3(im,options.sigma);
    subplot(2,2,2)
    imshow(im_smooth(:,:,floor(size(tmp.tmp,3)/2)),[])
    impixelinfo
    rangeMin  = options.rangeMin;
    rangeMax  = options.rangeMax;
    im_smooth   (im_smooth > rangeMax) = rangeMax;
    im_smooth   (im_smooth < rangeMin) = rangeMin;
    subplot     (2,2,3)
    imshow      (im_smooth(:,:,floor(size(tmp.tmp,3)/2)),[])
    impixelinfo
    %% Rescale the data to the range [0, 1].
    tmp   = (im_smooth - rangeMin) / (rangeMax - rangeMin);
    subplot     (2,2,4)
    imshow      (tmp(:,:,floor(size(tmp,3)/2)),[])
    impixelinfo
    %% Save files
    savename  = strcat('T', num2str(k), '.mat');
    save(savename, 'tmp', '-v7.3');
    pause(0.01)
end
