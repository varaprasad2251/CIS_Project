function ProcessImages(inputFolder, outputFolder)
    % inputFolder: Path to the folder containing input images
    % outputFolder: Path to the folder where processed images will be saved
    
    % Create output folder if it doesn't exist
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    % List all files and folders in the input folder
    contents = dir(inputFolder);
    
    % Iterate through each item in the input folder
    for i = 1:length(contents)
        item = contents(i);
        
        % Skip '.' and '..' directories
        if strcmp(item.name, '.') || strcmp(item.name, '..')
            continue;
        end
        
        % Construct full paths for the current item
        itemInputPath = fullfile(inputFolder, item.name);
        itemOutputPath = fullfile(outputFolder, item.name);
        
        % If the current item is a folder, recursively process its contents
        if item.isdir
            % Create corresponding subfolder in the output folder
            if ~exist(itemOutputPath, 'dir')
                mkdir(itemOutputPath);
            end
            
            % Recursively process the contents of the subfolder
            ProcessImages(itemInputPath, itemOutputPath);
        elseif endsWith(item.name, '.png') || endsWith(item.name, '.jpg')
            % Read and process the image
            cfa = imread(itemInputPath);
            % Process the image as per your requirements
            % Example processing code (assuming the code from your previous snippet):
            blue = cfa(2:2:end, 2:2:end);
            green1 = cfa(1:2:end, 2:2:end);
            red = cfa(1:2:end, 1:2:end);
            green2 = cfa(2:2:end, 1:2:end);
            rawCombined = cat(3, blue, green1, red, green2);
            out = single(rawCombined) / (4 * 255);
            networkInput = dlarray((out),"SSC");
            saveVarsMat = load('matlab.mat');

            net = saveVarsMat.net;
            output = forward(net,networkInput);
            image_data = extractdata(output);
            
            % Write the processed image to the output folder
            [~, name, ~] = fileparts(item.name);
            % outputFilename = fullfile(itemOutputPath, [name, '.png']);
            imwrite(image_data, itemOutputPath);
        end
    end
end
