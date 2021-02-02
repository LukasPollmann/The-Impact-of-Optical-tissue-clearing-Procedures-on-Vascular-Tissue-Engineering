clearvars
close all
clc

Labeled_images_dir = uigetdir(pwd, 'Where are the labeled Images located?');
cd (Labeled_images_dir)
files = dir('*.tif');

%% For rescaling between 0 and 1
rangeMin                = 0;  
rangeMax                = 255;
%% Set parameters & pre-allocation
parameters.redfac    = 1;
parameters.datatype  = 'label'; % label, whatever(for cropped images)
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

%% give the Name similar to the Raw data

x= 64; % Minimal dimensions for cropped images
y= 64; 
z= 64;

volReader  = @(x) matRead(x);
Labels     = pixelLabelDatastore(Labeled_images_dir,{'b', 'v'}, ...
             [0 1], 'FileExtensions','.mat','ReadFcn',volReader);
         

II   = zeros(length(Labels.Files),2);
hFig = figure;
for k = 1 : length(Labels.Files)
    tmp     = load(Labels.Files{k});
    disp    ([ 'Image' num2str(k)  '--size :' num2str(size(tmp.tmp))]);
    II(k,:) = [min(tmp.tmp(:)) max(tmp.tmp(:))];
    disp      ([ 'IntensityMin : ' num2str(II(k,1)) ', IntensityMax : ' ...
        num2str(II(k,2))]);
    if size(tmp.tmp,1)< x || size(tmp.tmp,2)< y || ...
            size(tmp.tmp,3)< z
        error('Error. \n image size must be => patch size')
    end
    imshow(tmp.tmp(:,:,floor(size(tmp.tmp,3)/2)),[])
    title(Labels.Files{k})
    disp(Labels.Files{k})
    hFig.WindowState = 'maximized';
    impixelinfo
    %% Rescale the data to the range [0, 1].
    tmp   = (tmp.tmp - rangeMin) / (rangeMax - rangeMin);
    imshow      (tmp(:,:,floor(size(tmp,3)/2)),[])
    title(Labels.Files{k})
    disp(Labels.Files{k})
    impixelinfo
    %% Save files
    savename  = strcat('T', num2str(k), '.mat');
    save(savename, 'tmp');
    pause(0.01)
end

