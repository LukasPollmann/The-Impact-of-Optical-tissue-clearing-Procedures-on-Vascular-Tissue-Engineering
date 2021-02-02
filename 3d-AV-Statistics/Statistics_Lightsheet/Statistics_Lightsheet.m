clearvars
close all
clc

 
Scaling_factor  = 0.25; %how much have you downsized the original data?

Predicted_labels_dir= uigetdir(pwd, 'Get predicted Labels directory');
cd (Predicted_labels_dir)
load predictedLabels.mat
load labelsName.mat

Labeled_images_dir = uigetdir(pwd, 'Get labeled images directory');
cd (Labeled_images_dir)
files = dir('*_labeled.tif');

Script_dir      = uigetdir(pwd, 'Where is the Script located?');
cd (Script_dir)
[fname, pname]  = uigetfile(pwd, 'Load Lookup table', '*.xlsx');
Datainformation = importfile([pname fname], "Tabelle1", [2, 10]);

Results_dir     = uigetdir(pwd, 'Where should Results been saved?');

%% Resolving predicted Labels names
n_idx1 = strfind(labelsName, '\');
n_idx2 = strfind(labelsName, '.mat');
name   = cell(size(labelsName));
for ID = 1 : size(n_idx1,1)
    name{ID}   = labelsName{ID}(n_idx1{ID}(end)+1:n_idx2{ID}(end)-1);
end

    %% Resolving Labeled Volume Names
    cd (Labeled_images_dir)
    LabeledVol = cell(1, length(files));
for k = 1: length(files)
    n_idx3   = numel('_labeled.tif');
    t        = struct2table (files);
for row = 1 : size(t, 1)
  % For every row...
  oldString   = t.name{row};
  t.name{row} = oldString(1:end-n_idx3);
end

    numimgs  = size(imfinfo(files(k).name),1);
    rawfile  = dir(files(k).name);
    rawData  = imread(rawfile.name);
    Vollabel = zeros(size(rawData,1),size(rawData,2), numimgs);
    for j = 1 : numimgs
        Vollabel(:,:,j) = imread(rawfile.name, j);
    end
    LabeledVol(k)    = {Vollabel}; 
end
%% Set parameters
nfeat = 3;
feats = zeros(length(numel(Datainformation.Mat_Name)),nfeat);

for idx = 1 : size(Datainformation,1)
    fname           = Datainformation.Name(idx);
    Labeled_number  = find(strcmp(t.name, fname) == 1);
    b               = find (Datainformation.Name ==fname);
    Mat_name        = Datainformation.Mat_Name (b);
    Predicted_number= extractAfter(Mat_name, 'T');
    Predicted_number= str2double (Predicted_number);
    voxelsize       = nthroot((1/Scaling_factor)^3*...
                      (Datainformation.x(idx)*Datainformation.y(idx)*...
                      Datainformation.z(idx)),3); %µm/pixel
%% Load predicted Volume
T = predictedLabels {Predicted_number};
T = double(T)>1;
TT= imresize3 (T, 0.25); 
T = imresize3 (TT, 4); % need to do that to combine with labeled Volume

% Get rid of small dots and junk by taking only the biggest 6-pixel
% connected volume called "Whole"

CC   = bwconncomp(T, 6);
S    = regionprops3(CC, 'Volume');
L    = labelmatrix(CC);
Whole= ismember(L, find([S.Volume] >= max(S.Volume)));

%% Extract the Main Vessel from predicted Data
T     = imdilate(T, strel('cuboid', [20 4 4]));
Ttmp  = imfill(T, 26, 'holes');
for i=1:size(Ttmp,3)
    Ttmp(:,:,i) = bwareaopen(Ttmp(:,:,i), 10000);
end

CC  = bwconncomp(Ttmp, 26);
S   = regionprops3(CC, 'Volume');
L   = labelmatrix(CC);
T   = ismember(L, find([S.Volume] >= max(S.Volume)));
se  = strel('sphere', 6);
Ero = imerode(T, se);  % remove small Vessels attached to the main Vessel
CC  = bwconncomp(Ero, 26);
S   = regionprops3(CC, 'Volume');
L   = labelmatrix(CC);
TT  = ismember(L, find([S.Volume] >= max(S.Volume))); %Main Vessel

Se = strel('sphere', 10); 
Td = imdilate(TT, Se); % fill holes
Td1= convn(Td, ones(9,9,9), 'same');
Td1= Td1 > 0;
for i=1:size(Td1,3)
    Td1(:,:,i) = imfill(Td1(:,:,i), 'holes');
end

Vessels= Whole-Td1; % get only the small Vessels
Vessels= Vessels>0;
se     = strel('sphere', 5);
Vesselsdilated=imdilate (Vessels, se); %Connect small Vessels by dilation

%% Now continue with your labeled volume
I  = LabeledVol{Labeled_number};
D  = imresize3 (I,0.25);
D  = bwmorph3(D, 'clean');
D  = bwmorph3(D, 'fill');

for i=1:size(D,3)
    D(:,:,i) = bwareaopen(D(:,:,i), 1000);
end

D  = imopen(D, strel('sphere', 3));
CC = bwconncomp(D);
S  = regionprops3(CC, 'Volume');
L  = labelmatrix(CC);
J  = ismember(L, find([S.Volume] >= max(S.Volume)));
J  = imdilate(J, strel('sphere', 20));
J  = J>0;
BW = bwskel(J);

% post proessing of skeleton
skel = imdilate(BW, strel('sphere', 20));
for i=1:size(skel,3)
    skel(:,:,i) = imfill(skel(:,:,i), 'holes');
end

skel_calculation = bwskel(skel); % second calculation
                                 % This skeleton will be used for Length
                                 % Calculation
skelt = imdilate(skel_calculation, strel('sphere', 20));
skelup= imresize3(skelt, 4);    %
CC    = bwconncomp(skel_calculation, 26);  

Length_vessel  = ((numel(CC.PixelIdxList{:})*4)*voxelsize)/1000; 
% Main Vessel Length in mm, first all connected Pixels were counted,
% multiplied with 4 because it was downsized in Line 114, then Voxelsize
% was multiplied (calculated before). At the End divided with 1000 to get
% the Length in mm insteadt of µm


%% Combine small Vessels and painted Main Vessel
Combined= Vesselsdilated|skelup;
CC      = bwconncomp(Combined, 26);
S       = regionprops3(CC, 'Volume');
L       = labelmatrix(CC);
Tree    = ismember(L, find([S.Volume] >= max(S.Volume)));
Combined= imresize3(Tree, 0.5);
SKEL    = bwskel(Combined);
skel    = imdilate(SKEL, strel('sphere', 5));
for i=1:size(skel,3)
    skel(:,:,i) = imfill(skel(:,:,i), 'holes');
end
skelt = bwskel(skel);
TREE= imdilate (skelt, strel( 'sphere', 3));
Bpoints= bwmorph3(skelt, 'branchpoints'); %Finding the branchpoints
Epoints= bwmorph3 (skelt, 'endpoints');   %Finding the Endopoints
Branchpoints= imdilate (Bpoints, strel('sphere', 3));
Number_Branchpoints=nnz(Bpoints==1);
Number_Endpoints=nnz(Epoints==1);

%% Display the result

% Diplay 3D data and save GIF
fname_tmp      = convertStringsToChars(fname);
gif_name       = [fname_tmp,'.gif'];
visScaleFactor = [1 1 2];
viewPnlPred    = uipanel(figure,'Title', ...
    'Skeleton /Branchpoints Overlay');
hP             = labelvolshow(categorical(Branchpoints, [0 1] ),TREE, ...
    'Parent', viewPnlPred, 'LabelColor', ...
    [0 0 0;1 0 0], 'VolumeThreshold',0.05, ...
    'LabelOpacity', [0.01; 0.18], ...
    'BackgroundColor', [0 0.9 0], ...
    'ScaleFactors', visScaleFactor);
set              (gcf, 'units','normalized','outerposition',[0 0 1 1]);


hP.VolumeOpacity      = 0.1;
hP.LabelVisibility(1) = 0;
CameraUpVector        = [-0.2 -1 -0.2];
CameraPosition        = [-2 -1 4];
vec                   = linspace(CameraPosition(1),3,100)';
xcam                  = vec;
ycam                  = CameraPosition(2)*ones(size(vec,1),1);
zcam                  = CameraPosition(3)*ones(size(vec,1),1);
myPosition            = [xcam ycam zcam];
hP.CameraUpVector     = CameraUpVector;
hP.CameraPosition     = CameraPosition;

cd (Results_dir)

% Video capturing/saving for predicted Skeleton
for id = 1:length(vec)
    hP.CameraPosition = myPosition(id,:);
    I                 = getframe(gcf);
    [indI,cm]         = rgb2ind(I.cdata,256);
    if id == 1
        imwrite(indI, cm, gif_name, 'gif', ...
            'Loopcount', inf, 'DelayTime', 0.01);
    else
        imwrite(indI, cm, gif_name, 'gif', 'WriteMode', ...
            'append', 'DelayTime', 0.01);
    end
end
disp (['Number_Endpoints : ', num2str(Number_Endpoints)]);
disp (['Number_Branchpoints : ', num2str(Number_Branchpoints)]);
Table_Variables= {'Length_of_the_labeled_Vessel_in_mm',...
                 'Number of Endpoints', 'Number of Branchpoints'};
Table_Rowname = fname;
Table = table(Length_vessel,Number_Endpoints,Number_Branchpoints, 'VariableNames', Table_Variables, 'Rownames',Table_Rowname);
Table = splitvars (Table);
if idx == 1
   writetable (Table,'Parts_results.xlsx', 'WriteMode', 'Append', 'WriteVariableNames', true, 'WriteRowNames', true)
else
    writetable (Table,'Parts_results.xlsx', 'WriteMode', 'Append', 'WriteVariableNames', false, 'WriteRowNames', true)
end
end
