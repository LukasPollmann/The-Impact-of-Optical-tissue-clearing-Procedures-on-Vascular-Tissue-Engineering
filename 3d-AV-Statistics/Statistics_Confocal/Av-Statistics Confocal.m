clearvars
close all
clc

load predictedLabels.mat
load groundTruthLabels.mat
load labelsName.mat

pname_rawImages = uigetdir(pwd, 'Get raw images directory');
[fname, pname]  = uigetfile(pwd, 'Load Lookup table', '*.xlsx');
Datainformation = importfile([pname fname], "Tabelle1", [2, 9]);

% Resolving names
n_idx1 = strfind(labelsName, '\');
n_idx2 = strfind(labelsName, '.mat');
name   = cell(size(labelsName));
for ID = 1 : size(n_idx1,1)
    name{ID}   = labelsName{ID}(n_idx1{ID}(end)+1:n_idx2{ID}(end)-1);
end

%% Set parameters
minVoxelSize = 0.5;
% Data type specifications
sfact = 0.25;
mtype = 'Confocal';
nfeat = 3;
dtype = 7;
idx   = find(Datainformation.Day == dtype);
Info  = Datainformation(idx,:);
Info  = Info(Info.Microscope == mtype,:);

if dtype == 28
    visScaleFactor = [1 1 1];
else
    visScaleFactor = [1 1 2];
end

feats = zeros(length(idx),nfeat);
 
%% Loop for vessel skeleton caulculation
for idx = 1 : length(idx)
    fname           = Info.Mat_Name(idx);
    p               = find(strcmp(name, fname) == 1);
    scalingfactor   = Info.Scalefactor_preprocessing(idx);
    voxelsizevol    = (1/str2double(scalingfactor))^3*...
                      (Info.x(idx)*Info.y(idx)*Info.z(idx));
    voxelsize       = nthroot((1/str2double(scalingfactor))^3*...
                      (Info.x(idx)*Info.y(idx)*Info.z(idx)),3);
                  
    % Load raw data (resize + smooth)             
    tmp             = load([pname_rawImages '\' char(fname) '.mat']);
    Io              = tmp.tmp;
    Io              = imresize3(Io, sfact);    
    Io              = imgaussfilt3(Io,1);
    
    % Load and check the predicted data
    II              = double(predictedLabels{p})>1;
    II              = bwmorph3(II, 'majority');
    II              = bwmorph3(II, 'clean');
    if dtype ~= 28
        II              = imdilate(II, strel('sphere', 3));
    end
    CC              = bwconncomp(II, 26);
    S               = regionprops3(CC, 'Volume');
    vols            = S.Volume.*voxelsizevol;    
    
    bigvols         = find(vols > minVoxelSize);
    LL              = labelmatrix(CC);
    
    % check if predictedData is bigger than the minVoxelSize
    if ~isempty(bigvols)
        I           = ismember(LL, find(vols >= minVoxelSize));
    else
        I           = double(predictedLabels{p})>1;
        I           = bwmorph3(I, 'clean');
    end
    
    % resize the predictedData / apply convolution
    I               = imresize3(I, sfact);
    II              = convn(I, ones(5,5,5), 'same');
    II              = II>0;
    
    if ~isempty(bigvols)
        II          = bwmorph3(II, 'majority');
        II          = bwmorph3(II, 'clean');
        II          = bwareaopen(II, 10000);
    end
    
    II              = imdilate(II, strel('sphere', 20));
    
    for i=1:size(II,3)
            II(:,:,i) = imfill(II(:,:,i), 'holes');
    end
    
    
    % Calculate the skeleton
    
    skel            = bwskel(II); % first calculation
    
    % post proessing of skeleton
    skel            = imdilate(skel, strel('sphere', 20));
    for i=1:size(skel,3)
        skel(:,:,i) = imfill(skel(:,:,i), 'holes');
    end
    
    skelt = bwskel(skel); % second calculation
    
   
    % calculate skeleton objects and take the longest part
    CC             = bwconncomp(skelt, 26);
    S              = regionprops3(CC, 'Volume');
    L              = labelmatrix(CC);
    skelt          = ismember(L, find([S.Volume] >= max(S.Volume)));
    
    % remove small spurious branches
    SE             = strel('sphere',3);
    E              = bwmorph3(skelt, 'endpoints');
    B              = imdilate(bwmorph3(skelt, 'branchpoints'),SE);
    B(E|~skelt)    = false;
    skelpart       = skelt-B;
    skelpart       = imdilate(skelpart, strel('sphere', 9));
    
    skelt          = bwskel(logical(skelpart)); % third calculation
    E              = bwmorph3(skelt, 'endpoints');
    idxE           = find(E);
    npaths         = idxE -1;
    B              = imdilate(bwmorph3(skelt, 'branchpoints'),SE);
    B(E|~skelt)    = false;
    skelpart       = skelt-B;
    
    skelt          = bwskel(logical(skelpart)); % fourth calculation
    
   
    % dilate/ the skeleton for 
    CC             = bwconncomp(skelt, 26);
    skelt          = imdilate(skelt, strel('sphere', 5));
%     figure;          volshow(skelpart,'ScaleFactors', visScaleFactor);
     
    % calculate vessel length
    if CC.NumObjects == 1
        vessel_length  = ((numel(CC.PixelIdxList{:}))*voxelsize) ...
                          *(1/sfact);
        disp             (['Length  : ' ...
        num2str(vessel_length) ' mm'])
    else
        for no = 1 : CC.NumObjects
            vessel_length(no) = ((numel(CC.PixelIdxList{no}))*voxelsize)...
                                *(1/sfact);
            vessel_length     = max(sum(nchoosek(vessel_length,2),2));
        end
     
       disp                    (['Length : ' num2str(vessel_length) ' mm'])
    end
    
    % calculate volume and principal axis length
    V              = I|skelt;  %combine skeleton and rescaled step-1 vol
    V              = imdilate(V, strel('sphere', 3));
    V              = bwmorph3(V, 'fill');
    for i=1:size(V,3)
        V(:,:,i)   = imfill(V(:,:,i), 'holes');
    end
    
    VCC            = bwconncomp(V, 26);
    S              = regionprops3(VCC, 'Volume');
    L              = labelmatrix(VCC);
    V              = ismember(L, find([S.Volume] >= max(S.Volume)));
    VCC            = bwconncomp(V, 26);
    S              = regionprops3(VCC, 'Volume', 'PrincipalAxisLength');
    vessellength3D = max(S.PrincipalAxisLength) * voxelsize *(1/sfact);
    vessel_volume  = S.Volume*voxelsizevol*(1/sfact)^3;

    
    disp             (['Volume  : ' num2str(vessel_volume) ' µL'])
    disp             (['Principal Axis Length  : ' ...
                      num2str(vessellength3D) ' mm'])
      
    feats(idx,:)   = [vessel_volume vessellength3D vessel_length];
 
    
    %% Diplay 3D data and save GIF
    viewPnlPred    = uipanel(figure,'Title', ...
                     'Prediction/Skeleton Overlay');
    hP             = labelvolshow(categorical(skelt, [0 1] ),Io, ...
                     'Parent', viewPnlPred, 'LabelColor', ...
                     [0 0 0;1 0 0], 'VolumeThreshold',0.05, ...
                     'LabelOpacity', [0.01; 0.18], ...
                     'BackgroundColor', [0 0.9 0], ...
                     'ScaleFactors', visScaleFactor);
    set              (gcf, 'units','normalized','outerposition',[0 0 1 1]);
    
    
    hP.VolumeOpacity      = 0.1;
    hP.LabelVisibility(1) = 0;
    filename              = ['Skel_' name{p} '_' mtype ...
                              '_d' num2str(dtype) '.gif'];
    CameraUpVector        = [-0.2 -1 -0.2];
    CameraPosition        = [-2 -1 4];
    vec                   = linspace(CameraPosition(1),3,100)';
    xcam                  = vec;
    ycam                  = CameraPosition(2)*ones(size(vec,1),1);
    zcam                  = CameraPosition(3)*ones(size(vec,1),1);
    myPosition            = [xcam ycam zcam];
    hP.CameraUpVector     = CameraUpVector;
    hP.CameraPosition     = CameraPosition;
    
    % Video capturing/saving for predicted Skeleton
    for id = 1:length(vec)
        hP.CameraPosition = myPosition(id,:);
        I                 = getframe(gcf);
        [indI,cm]         = rgb2ind(I.cdata,256);
        if id == 1
            imwrite(indI, cm, filename, 'gif', ...
                'Loopcount', inf, 'DelayTime', 0.01);
        else
            imwrite(indI, cm, filename, 'gif', 'WriteMode', ...
                'append', 'DelayTime', 0.01);
        end
    end
    
end

% Save vessel features to an excel file
feats_names  = {'Volume(microL)', 'PrincipalAxisLength(mm)', 'Length(mm)'};
T            = array2table(feats, 'VariableNames',feats_names, ...
               'RowNames', Info.Mat_Name);
writetable     (T,[pwd '\' 'Vessel_features_d' num2str(dtype) '.csv'], ...
                'WriteRowNames',true)


