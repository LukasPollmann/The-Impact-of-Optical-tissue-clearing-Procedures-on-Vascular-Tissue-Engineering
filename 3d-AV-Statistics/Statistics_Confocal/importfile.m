function Datainformation1 = importfile(workbookFile, sheetName, dataLines)
%IMPORTFILE Import data from a spreadsheet
%  DATAINFORMATION1 = IMPORTFILE(FILE) reads data from the first
%  worksheet in the Microsoft Excel spreadsheet file named FILE.
%  Returns the data as a table.
%
%  DATAINFORMATION1 = IMPORTFILE(FILE, SHEET) reads from the specified
%  worksheet.
%
%  DATAINFORMATION1 = IMPORTFILE(FILE, SHEET, DATALINES) reads from the
%  specified worksheet for the specified row interval(s). Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  Datainformation1 = importfile("C:\Users\Nici & Luki\Documents\confocal prediction\Data_information.xlsx", "Tabelle1", [2, 9]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 27-Jun-2020 17:13:32

%% Input handling

% If no sheet is specified, read first sheet
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% If row start and end points are not specified, define defaults
if nargin <= 2
    dataLines = [2, 9];
end

%% Setup the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 9);

% Specify sheet and range
opts.Sheet = sheetName;
opts.DataRange = "A" + dataLines(1, 1) + ":I" + dataLines(1, 2);

% Specify column names and types
opts.VariableNames = ["Name", "Mat_Name", "x", "y", "z", "Microscope", "Day", "Scalefactor_preprocessing", "Unit_Voxelsize"];
opts.VariableTypes = ["string", "string", "double", "double", "double", "categorical", "double", "string", "categorical"];

% Specify variable properties
opts = setvaropts(opts, ["Name", "Mat_Name", "Scalefactor_preprocessing"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Name", "Mat_Name", "Microscope", "Scalefactor_preprocessing", "Unit_Voxelsize"], "EmptyFieldRule", "auto");

% Import the data
Datainformation1 = readtable(workbookFile, opts, "UseExcel", false);

for idx = 2:size(dataLines, 1)
    opts.DataRange = "A" + dataLines(idx, 1) + ":I" + dataLines(idx, 2);
    tb = readtable(workbookFile, opts, "UseExcel", false);
    Datainformation1 = [Datainformation1; tb]; %#ok<AGROW>
end

end