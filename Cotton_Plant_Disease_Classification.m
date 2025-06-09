clear all;
clc;

%Below lines of code to get the dataset from remote repo
url = 'https://github.com/zubairAhmed777/App_of_DataScience/archive/refs/heads/main.zip';
disp(['Downloading Dataset from : ', url]);
opts = weboptions('Timeout', 60, 'CertificateFilename', ''); % Adjust timeout if necessary
% Downloading the file with try-catch block
try
    outputFile = 'App_of_DataScience.zip';
    websave(outputFile, url, opts);
    unzip(outputFile, pwd);
    disp('Repository downloaded and extracted successfully.');
catch ME
    disp(['Failed to download the dataset. Error: ', ME.message]);
end


disp('--------------------START Feature Extraction-------------------------')

% Define the directory path
folderPath = 'App_of_DataScience-main/';
%Feature Extraction
all_extracted_glcm_features = extract_features(folderPath);
temp_data = all_extracted_glcm_features;

% Setting a random seed for reproducibility
seedValue = 3; 
rng(seedValue);

% Generate a random permutation of row indices
numRows = size(all_extracted_glcm_features, 1);
randomOrder = randperm(numRows);

% Shuffle the matrix rows
all_extracted_glcm_features = all_extracted_glcm_features(randomOrder, :);

%Splitting the dataset in 80:20 for tarin and test 
split_row = floor(size(all_extracted_glcm_features,1) * 0.80);

%getting train data set
train_data = all_extracted_glcm_features(1:split_row, :);
%Getting test dataset
test_data = all_extracted_glcm_features(split_row+1:end, :);

disp('--------------------DONE Feature Extraction-------------------------')

%Top level function to loop over image directories
function all_glcm_features = extract_features(folderPath)
    
    % Get the list of files and subdirectories
    contents = dir(folderPath);
    % removing (.)--> current directory
    contents = contents(~ismember({contents.name}, {'.', '..'}));
    
    % Vector to store the features
    all_glcm_features = [];
    % Looping over each directory
    for i = 1:length(contents)
        % Creating the path of sub directory
        path = [folderPath, contents(i).name, '/'];
        % Getting class name and label for the respective directory
        [class_name, class_label] = get_class_label(contents(i).name);
        disp(class_name);
        disp(class_label);
        % get the images in directory
        images = dir(path);
        % Filter for files with .jpg, .png, and .jpeg extensions
        images = images(~ismember({images.name}, {'.', '..'})); % Remove '.' and '..'
        images = images(~[images.isdir]);
        images = images(endsWith({images.name}, {'.jpg', '.jpeg'}, 'IgnoreCase', true));
        % Looping over all the images in the directory to extract features
        for j=1:length(images)
            image = [folderPath, contents(i).name, '/', images(j).name];
            image_path = image;
            image = imread(image); % Reading image data
            disp(image_path);
            glcm_features = extract_glcm_features(image, class_label);
            all_glcm_features = [all_glcm_features; glcm_features];
        end
    end
end

function features = extract_glcm_features(image, class_label)  
    % Preprocessing
    % Resizing the image
    img_resized = imresize(image, [1024, 1024]);
    %img_resized = bilinear_interpolation_color(image, 1024, 1024);
    % Noise removal using median filter
    if ndims(img_resized) == 3
        % Separate the color channels
        redChannel = img_resized(:, :, 1);
        greenChannel = img_resized(:, :, 2);
        blueChannel = img_resized(:, :, 3);
        filter_size = [3,3];
        % Apply median filtering to each channel        
        filteredRed = customMedianFilter2(redChannel);
        filteredGreen = customMedianFilter2(greenChannel);
        filteredBlue = customMedianFilter2(blueChannel);
    
        %Combine the filtered channels back into a color image
        filtered_img = cat(3, filteredRed, filteredGreen, filteredBlue);
    end
    % Contrast enhancemennt of image
    filtered_img = customImadjustExact(filtered_img, customStretchlimExact(filtered_img));
    % RGB to HSV color space 
    hsv = customRgb2Hsv(filtered_img);
    h = hsv(:, :, 1); % Hue channel
    s = hsv(:, :, 2); % Saturation channel
    v = hsv(:, :, 3); % Value channel
    % Colour features extraction
    color_features = [mean(h(:)), std(h(:)), mean(s(:)), std(s(:)), mean(v(:)), std(v(:))];
    % Thresholding on saturation to segment disease regions
    threshold = max(s(:)) * 0.099; % Adaptive thresholding on saturation
    mask = s > threshold;
    
    % Apply additional thresholding on the Hue channel
    Inew = h .* mask; % Combining initial mask with HUe channel
    mask2 = (Inew > 0.25) & (Inew < 0.417);
    
    %-------------------- Morphological Operations -------------------------------
    % Clean up the mask using morphological operations
    %se1 = strel('square', 3); % Structuring element for dilation
    se1 = customStrelSquare(3);
    %dilatedMask = imdilate(mask2, se1); % Dilation
    dilatedMask = customImdilate(mask2, se1);
    %se2 = strel('square', 3); % Structuring element for erosion
    se2 = customStrelSquare(3);
    %cleanedMask = imerode(dilatedMask, se2); % Erosion
    cleanedMask = customImerode(dilatedMask, se2);

    %-------------------- Blob Detection -------------------------------
    % Extract the smallest blobs
    numBlobs = 100; % Get Number of smallest blobs to extract
    maxbolb = 0;
    if 1 == class_label
        maxbolb = 1;
    end
    blobsMask = extractBlobs(cleanedMask, numBlobs, maxbolb); % Blob extraction
    
    %-------------------- Converting Image Back to RGB -------------------------------
    % Create masked HSV image
    masked_hue = h .* ~blobsMask;
    masked_saturation = s .* ~blobsMask;
    masked_value = v .* ~blobsMask;
    
    % Combine masked HSV channels
    masked_hsv = cat(3, masked_hue, masked_saturation, masked_value);
    
    masked_rgb = customHsv2Rgb(masked_hsv); % Convert back to RGB
    
    %--------------------again  applying morphological operations (optional)------------------------------- 
    a=strel('square',2);
    f4=imdilate(masked_rgb,a);
    se = strel('rectangle',[2 1]);        
    erodedBW2 = imerode(f4,se);
    seg_img = erodedBW2;

    %---------------------------------------------color feature extraction, GLCM Feature extraction --------------------------------------
    if ndims(seg_img) == 3
        % Converting RGB image to Gray image
        enhanced_img = customRgb2Gray(seg_img);
    end
    
    %  GLCM for Texture features extraction
    glcms = graycomatrix(enhanced_img, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    % Deriving Statistics from GLCM
    stats = customGraycoprops(glcms);
    Contrast = stats.Contrast;
    Correlation = stats.Correlation;
    Energy = stats.Energy;
    Homogeneity = stats.Homogeneity;

    Mean = customMean2(enhanced_img);
    Standard_Deviation = customStd2(enhanced_img);
    Entropy = customEntropy(enhanced_img);
    RMS = customRMS(enhanced_img);
    Variance = customVariance(enhanced_img);
    Kurtosis = customKurtosis(enhanced_img);
    Skewness = customSkewness(enhanced_img);
    % combining all the statistical features
    texture_features = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Kurtosis, Skewness];
        
    % Combine all statisctical features, colour and glcm features
    features = [color_features, texture_features, class_label];
end

% Function for extracting smallest blobs
function binaryImage = extractBlobs(binaryImage, numToextract, maxblob)
    % Check if binaryImage is logical, if not, binarize it
    if ~islogical(binaryImage)
        binaryImage = imbinarize(binaryImage); 
    end

    % Label connected components in the binary image
    [labeledImage, numberOfBlobs] = bwlabel(binaryImage, 8);

    % If there are no blobs, return an empty mask
    if numberOfBlobs == 0
        disp('No blobs detected.');
        binaryImage = false(size(binaryImage));
        return;
    end

    % Measure blob areas
    blobMeasurements = regionprops(labeledImage, 'Area');
    allAreas = [blobMeasurements.Area]; % Extract all blob areas

    % Sort areas in ascending order and get indices 
    if maxblob == 1
        [~, sortIndexes] = sort(allAreas, 'descend');
        % Adjusting number To Extract if it exceeds the number of blobs
        numToextract = min(numToextract, numel(sortIndexes));
        small_blobs = sortIndexes(1:max(numToextract, numel(sortIndexes)));
    else
        [~, sortIndexes] = sort(allAreas, 'ascend');
        % Adjusting number To Extract if it exceeds the number of blobs
        % limit
        numToextract = min(numToextract, numel(sortIndexes));
        small_blobs = sortIndexes(1:min(numToextract, numel(sortIndexes)));
    end

    % Create a binary mask for the smallest blobs
    binaryImage = ismember(labeledImage, small_blobs);

    % Convert from labeled image to binary
    binaryImage = binaryImage > 0; 
end


function [class_name, class_label] = get_class_label(name)
    % Define keys and values
    keys = { 'Army worm', 'Healthy leaf', 'Powdery Mildew'};
    values = [1, 2, 3]; % Numerical labels for the classes

    % Create the dictionary
    classDict = containers.Map(keys, values);
    
    if strcmp(name, 'Army worm')
        class_name = 'Army worm';
        class_label = classDict(class_name);
    end
    if strcmp(name, 'Healthy')
        class_name = 'Healthy leaf';
        class_label = classDict(class_name);
    end
    if strcmp(name, 'Powdery Mildew')
        class_name = 'Powdery Mildew';
        class_label = classDict(class_name);
    end
end

function resizedImage = bilinear_interpolation_color(inputImage, newRows, newCols)
    % Optimized bilinear interpolation for resizing images (color or grayscale).

    % Get the original dimensions
    [numRows, numCols, numChannels] = size(inputImage);

    % Precompute scale factors
    rowScale = numRows / newRows;
    colScale = numCols / newCols;

    % Compute the coordinates in the original image
    [newColIdx, newRowIdx] = meshgrid(1:newCols, 1:newRows);
    originalX = ((newColIdx - 0.5) * colScale) + 0.5; % Map to original column indices
    originalY = ((newRowIdx - 0.5) * rowScale) + 0.5; % Map to original row indices

    % Find the surrounding points
    xLower = floor(originalX); xUpper = ceil(originalX);
    yLower = floor(originalY); yUpper = ceil(originalY);

    % Handle edge cases (ensure indices are within bounds)
    xLower = max(1, min(xLower, numCols));
    xUpper = max(1, min(xUpper, numCols));
    yLower = max(1, min(yLower, numRows));
    yUpper = max(1, min(yUpper, numRows));

    % Compute interpolation weights
    xWeight = originalX - xLower;
    yWeight = originalY - yLower;

    % Initialize the resized image
    resizedImage = zeros(newRows, newCols, numChannels, 'like', inputImage);

    % Perform bilinear interpolation for each channel
    for channel = 1:numChannels
        % Extract the current channel
        channelData = double(inputImage(:, :, channel));

        % Perform interpolation
        interpolatedValues = ...
            (1 - xWeight) .* (1 - yWeight) .* channelData(sub2ind([numRows, numCols], yLower, xLower)) + ...
            xWeight .* (1 - yWeight) .* channelData(sub2ind([numRows, numCols], yLower, xUpper)) + ...
            (1 - xWeight) .* yWeight .* channelData(sub2ind([numRows, numCols], yUpper, xLower)) + ...
            xWeight .* yWeight .* channelData(sub2ind([numRows, numCols], yUpper, xUpper));

        % Assign to the resized image
        resizedImage(:, :, channel) = uint8(round(interpolatedValues));
    end
end

function outputImage = customMedianFilter2(inputImage)
    % Custom 3x3 median filter for 2D grayscale images
    % inputImage: Grayscale image to be filtered
    
    % Ensure the input is 2D (grayscale image)
    if ndims(inputImage) ~= 2
        error('Input must be a 2D grayscale image.');
    end

    % Get the size of the input image
    [numRows, numCols] = size(inputImage);

    % Pad the image with a 1-pixel symmetric border
    paddedImage = customSymmetricPad(inputImage, [1, 1]);

    % Initialize the output image
    outputImage = zeros(numRows, numCols, 'like', inputImage);

    % Apply the 3x3 median filter
    for i = 1:numRows
        for j = 1:numCols
            % Extract the 3x3 neighborhood
            neighborhood = paddedImage(i:i+2, j:j+2);
            
            % Compute the median of the neighborhood
            outputImage(i, j) = median(neighborhood(:));
        end
    end
end

function paddedImage = customSymmetricPad(inputImage, padSize)
    % Custom symmetric padding for a 2D image
    % inputImage: 2D image to be padded
    % padSize: [padRows, padCols] specifies the padding size for rows and columns
    
    % Extract the padding dimensions
    padRows = padSize(1);
    padCols = padSize(2);

    % Get the size of the original image
    [numRows, numCols] = size(inputImage);

    % Initialize the padded image
    paddedImage = zeros(numRows + 2 * padRows, numCols + 2 * padCols, 'like', inputImage);

    % Copy the original image into the center
    paddedImage(padRows+1:end-padRows, padCols+1:end-padCols) = inputImage;

    % Create symmetric padding for rows
    for i = 1:padRows
        paddedImage(padRows+1-i, padCols+1:end-padCols) = inputImage(i, :); % Top rows
        paddedImage(end-padRows+i, padCols+1:end-padCols) = inputImage(end-i+1, :); % Bottom rows
    end

    % Create symmetric padding for columns
    for j = 1:padCols
        paddedImage(:, padCols+1-j) = paddedImage(:, padCols+j); % Left columns
        paddedImage(:, end-padCols+j) = paddedImage(:, end-padCols-j+1); % Right columns
    end

    % Padding in the corners
    for i = 1:padRows
        for j = 1:padCols
            % Top-left corner padding
            paddedImage(padRows+1-i, padCols+1-j) = paddedImage(padRows+1-i, padCols+1+j);
            % Top-right corner padding
            paddedImage(padRows+1-i, end-padCols+j) = paddedImage(padRows+1-i, end-padCols-j);
            % Bottom-left corner padding
            paddedImage(end-padRows+i, padCols+1-j) = paddedImage(end-padRows+i, padCols+1+j);
            % Bottom-right corner padding
            paddedImage(end-padRows+i, end-padCols+j) = paddedImage(end-padRows+i, end-padCols-j);
        end
    end
end

function outputImage = customImadjustExact(inputImage, inRange)
    % Custom implementation of MATLAB's imadjust
    % inputImage: Grayscale or RGB image
    
    % Determine the output range based on the input image's data type
    if isfloat(inputImage)
        outRange = [0, 1]; % Full range for float types
    elseif isinteger(inputImage)
        outRange = [0, double(intmax(class(inputImage)))]; % Full range for integer types
    else
        error('Unsupported image data type.');
    end

    % Handle grayscale or color images
    if ndims(inputImage) == 2
        % Grayscale image
        outputImage = adjustChannel(inputImage, inRange, outRange);
    elseif ndims(inputImage) == 3
        % RGB image: Process each channel independently
        [rows, cols, channels] = size(inputImage);
        outputImage = zeros(rows, cols, channels, 'like', inputImage);

        for channel = 1:channels
            outputImage(:, :, channel) = adjustChannel(...
                inputImage(:, :, channel), inRange(channel, :), outRange);
        end
    else
        error('Input must be a 2D grayscale or 3D RGB image.');
    end
end

function channelOut = adjustChannel(channelIn, inRange, outRange)
    % Adjust a single channel
    channelIn = double(channelIn);

    % Normalize input range and clip
    channelNorm = (channelIn - inRange(1)) / (inRange(2) - inRange(1));
    channelNorm = max(0, min(1, channelNorm)); % Clip to [0, 1]

    % Scale to output range
    channelOut = channelNorm * (outRange(2) - outRange(1)) + outRange(1);

    % Cast back to the original type
    channelOut = cast(channelOut, 'like', channelIn);
end


function inRange = customStretchlimExact(inputImage)
    % Custom implementation of MATLAB's stretchlim
    % inputImage: Grayscale or RGB image
    % Automatically computes stretch limits for [0.01, 0.99] percentiles

    % Defining Default percentiles
    lowerPercent = 0.01;
    upperPercent = 0.99;

    % Handle grayscale or color images
    if ndims(inputImage) == 2
        % Grayscale image
        inRange = calculateStretchlim(inputImage, lowerPercent, upperPercent);
    elseif ndims(inputImage) == 3
        % RGB image: Process each channel independently
        inRange = zeros(3, 2);
        for channel = 1:3
            inRange(channel, :) = calculateStretchlim(inputImage(:, :, channel), lowerPercent, upperPercent);
        end
    else
        error('Input must be a 2D grayscale or 3D RGB image.');
    end
end

function inRange = calculateStretchlim(channel, lowerPercent, upperPercent)
    % Calculate stretch limits for a single channel
    pixelValues = double(channel(:));
    pixelValues = sort(pixelValues);
    numPixels = numel(pixelValues);
    lowIndex = max(1, round(lowerPercent * numPixels));
    highIndex = min(numPixels, round(upperPercent * numPixels));
    inRange = [pixelValues(lowIndex), pixelValues(highIndex)];
end

function hsvImage = customRgb2Hsv(rgbImage)
    % Custom implementation of MATLAB's rgb2hsv
    % rgbImage: Input RGB image (values should be in the range [0, 1] or [0, 255])
    
    % Convert input to double for calculations
    if isinteger(rgbImage)
        rgbImage = double(rgbImage) / 255; % Normalize to [0, 1]
    end

    % Extract RGB channels
    R = rgbImage(:, :, 1);
    G = rgbImage(:, :, 2);
    B = rgbImage(:, :, 3);

    % Compute the max and min values across the channels
    Cmax = max(rgbImage, [], 3);
    Cmin = min(rgbImage, [], 3);
    delta = Cmax - Cmin;

    % Initialize HSV channels
    H = zeros(size(Cmax));
    S = zeros(size(Cmax));
    V = Cmax;

    % Compute Hue (H)
    idx = (delta > 0);
    % Red is max
    redIdx = idx & (Cmax == R);
    H(redIdx) = mod((G(redIdx) - B(redIdx)) ./ delta(redIdx), 6);
    % Green is max
    greenIdx = idx & (Cmax == G);
    H(greenIdx) = ((B(greenIdx) - R(greenIdx)) ./ delta(greenIdx)) + 2;
    % Blue is max
    blueIdx = idx & (Cmax == B);
    H(blueIdx) = ((R(blueIdx) - G(blueIdx)) ./ delta(blueIdx)) + 4;
    % Scale H to [0, 1]
    H = H / 6;
    H(H < 0) = H(H < 0) + 1;

    % Compute Saturation (S)
    S(Cmax > 0) = delta(Cmax > 0) ./ Cmax(Cmax > 0);

    % Combine HSV channels into a single image
    hsvImage = cat(3, H, S, V);
end

function structuringElement = customStrelSquare(size)
    % Custom implementation of strel('square', size)
    % size: Size of the square structuring element (positive integer)
    
    % Ensure size is a positive integer
    if ~isscalar(size) || size <= 0 || mod(size, 1) ~= 0
        error('Size must be a positive integer.');
    end

    % Create a square structuring element
    structuringElement = ones(size, size, 'logical');
end

function dilatedImage = customImdilate(binaryImage, structuringElement)
    % Custom implementation of imdilate for binary images
    % binaryImage: Input binary image (logical matrix)
    % structuringElement: Structuring element (logical matrix)

    % Validate inputs
    if ~islogical(binaryImage)
        error('Input image must be binary (logical).');
    end
    if ~islogical(structuringElement)
        structuringElement = logical(structuringElement);
    end

    % Get dimensions of the structuring element
    [seRows, seCols] = size(structuringElement);

    % Calculate padding size for the structuring element
    padRows = floor(seRows / 2);
    padCols = floor(seCols / 2);

    % Apply custom symmetric padding to the input binary image
    paddedImage = customSymmetricPad(binaryImage, [padRows, padCols]);

    % Initialize the output image
    [numRows, numCols] = size(binaryImage);
    dilatedImage = false(numRows, numCols);

    % Perform dilation
    for i = 1:numRows
        for j = 1:numCols
            % Extract the neighborhood region
            neighborhood = paddedImage(i:i+seRows-1, j:j+seCols-1);
            
            % Apply the structuring element (logical AND + OR operation)
            dilatedImage(i, j) = any(neighborhood(structuringElement));
        end
    end
end

function erodedImage = customImerode(binaryImage, structuringElement)
    % Custom implementation of imerode for binary images
    % binaryImage: Input binary image (logical matrix)
    % structuringElement: Structuring element (logical matrix)

    % Validate inputs
    if ~islogical(binaryImage)
        error('Input image must be binary (logical).');
    end
    if ~islogical(structuringElement)
        structuringElement = logical(structuringElement);
    end

    % Get dimensions of the structuring element
    [seRows, seCols] = size(structuringElement);

    % Calculate padding size for the structuring element
    padRows = floor(seRows / 2);
    padCols = floor(seCols / 2);

    % Apply custom symmetric padding to the input binary image
    paddedImage = customSymmetricPad(binaryImage, [padRows, padCols]);

    % Initialize the output image
    [numRows, numCols] = size(binaryImage);
    erodedImage = true(numRows, numCols);

    % Perform erosion
    for i = 1:numRows
        for j = 1:numCols
            % Extract the neighborhood region
            neighborhood = paddedImage(i:i+seRows-1, j:j+seCols-1);
            
            % Apply the structuring element (logical AND operation)
            erodedImage(i, j) = all(neighborhood(structuringElement));
        end
    end
end

function rgbImage = customHsv2Rgb(hsvImage)
    % Custom implementation of MATLAB's hsv2rgb
    % hsvImage: Input HSV image (values should be in the range [0, 1])

    % Extract HSV channels
    H = hsvImage(:, :, 1); % Hue
    S = hsvImage(:, :, 2); % Saturation
    V = hsvImage(:, :, 3); % Value

    % Initialize RGB channels
    R = zeros(size(H));
    G = zeros(size(H));
    B = zeros(size(H));

    % Compute RGB based on the HSV values
    C = V .* S; % Chroma
    X = C .* (1 - abs(mod(H * 6, 2) - 1)); % Second largest component
    m = V - C; % Adjustment factor

    % Hue ranges
    H = H * 6; % Scale H to [0, 6]
    idx1 = (H >= 0 & H < 1);
    idx2 = (H >= 1 & H < 2);
    idx3 = (H >= 2 & H < 3);
    idx4 = (H >= 3 & H < 4);
    idx5 = (H >= 4 & H < 5);
    idx6 = (H >= 5 & H <= 6);

    % Assign RGB values based on hue ranges
    R(idx1) = C(idx1); G(idx1) = X(idx1); B(idx1) = 0;
    R(idx2) = X(idx2); G(idx2) = C(idx2); B(idx2) = 0;
    R(idx3) = 0; G(idx3) = C(idx3); B(idx3) = X(idx3);
    R(idx4) = 0; G(idx4) = X(idx4); B(idx4) = C(idx4);
    R(idx5) = X(idx5); G(idx5) = 0; B(idx5) = C(idx5);
    R(idx6) = C(idx6); G(idx6) = 0; B(idx6) = X(idx6);

    % Add the adjustment factor to all channels
    R = R + m;
    G = G + m;
    B = B + m;

    % Combine into an RGB image
    rgbImage = cat(3, R, G, B);
end

function grayImage = customRgb2Gray(rgbImage)
    % Custom implementation of MATLAB's rgb2gray
    % rgbImage: Input RGB image (values in the range [0, 1] or [0, 255])

    % Ensure the input image is RGB
    if ndims(rgbImage) ~= 3 || size(rgbImage, 3) ~= 3
        error('Input must be a 3D RGB image.');
    end

    % Convert input to double for calculation
    if isinteger(rgbImage)
        rgbImage = double(rgbImage) / 255; % Normalize to [0, 1]
    end

    % Use the standard luminosity method for grayscale conversion
    % Grayscale intensity = 0.2989*R + 0.5870*G + 0.1140*B
    R = rgbImage(:, :, 1);
    G = rgbImage(:, :, 2);
    B = rgbImage(:, :, 3);
    grayImage = 0.2989 * R + 0.5870 * G + 0.1140 * B;

    % Scale back to the original type if necessary
    if isinteger(rgbImage)
        grayImage = uint8(grayImage * 255);
    end
end

function stats = customGraycoprops(glcm)
    % Compute texture properties from the GLCM
    % glcm: 3D array of size numLevels x numLevels x numOffsets

    % Initialize the stats structure
    stats.Contrast = [];
    stats.Correlation = [];
    stats.Energy = [];
    stats.Homogeneity = [];

    % Iterate over each GLCM (for each offset)
    for k = 1:size(glcm, 3)
        P = glcm(:, :, k); % Extract the GLCM for the current offset

        % Normalize the GLCM
        P = P / sum(P(:)); % Ensure it represents probabilities

        % Indices of the GLCM
        [I, J] = meshgrid(1:size(P, 1), 1:size(P, 2));

        % Mean and standard deviation
        mu_i = sum(I(:) .* P(:));
        mu_j = sum(J(:) .* P(:));
        sigma_i = sqrt(sum(((I(:) - mu_i).^2) .* P(:)));
        sigma_j = sqrt(sum(((J(:) - mu_j).^2) .* P(:)));

        % Contrast
        contrast = sum((I(:) - J(:)).^2 .* P(:));

        % Correlation
        if sigma_i > 0 && sigma_j > 0
            correlation = sum(((I(:) - mu_i) .* (J(:) - mu_j)) .* P(:)) / (sigma_i * sigma_j);
        else
            correlation = 0; % Avoid division by zero
        end

        % Energy
        energy = sum(P(:).^2);

        % Homogeneity
        homogeneity = sum(P(:) ./ (1 + abs(I(:) - J(:))));

        % Append results to the stats structure
        stats.Contrast = [stats.Contrast, contrast];
        stats.Correlation = [stats.Correlation, correlation];
        stats.Energy = [stats.Energy, energy];
        stats.Homogeneity = [stats.Homogeneity, homogeneity];
    end
end

% Custom mean2 implementation
function m = customMean2(image)
    m = sum(image(:)) / numel(image);
end

% Custom std2 implementation
function s = customStd2(image)
    m = customMean2(image);
    s = sqrt(sum((image(:) - m).^2) / (numel(image) - 1)); % Use n-1 for normalization
end

% Custom entropy implementation
function e = customEntropy(image)
    % Compute the histogram (normalized probabilities)
    histogram = histcounts(image(:), 256, 'Normalization', 'probability');
    % Remove zero entries to avoid log(0)
    histogram(histogram == 0) = [];
    e = -sum(histogram .* log2(histogram));
end

% Custom variance implementation
function v = customVariance(image)
    m = customMean2(image);
    v = sum((image(:) - m).^2) / (numel(image)); % Use n-1 for normalization
end

% Custom kurtosis implementation
function k = customKurtosis(image)
    m = customMean2(image);
    v = customVariance(image);
    normalizedData = (image(:) - m) / sqrt(v);
    k = mean(normalizedData.^4) - 3;
end

% Custom skewness implementation
function s = customSkewness(image)
    m = customMean2(image);
    v = customVariance(image);
    normalizedData = (image(:) - m) / sqrt(v);
    s = mean(normalizedData.^3);
end

function rms = customRMS(image)
    % Ensure the image is in double format
    image = double(image);    
    % Calculate the square of each pixel value
    squared_image = image.^2;    
    % Calculate the mean of the squared values
    mean_squared = mean(squared_image(:));    
    % Take the square root of the mean
    rms = sqrt(mean_squared);
end
function result = customMean(columnData)
    % Calculate the sum of the data
    totalSum = sum(columnData);

    % Count the number of elements
    numElements = length(columnData);

    % Compute the mean
    result = totalSum / numElements;
end
% Descriptive Statisctics
rows_with_last_column_2 = temp_data(temp_data(:, end) == 2, :);
rows_with_last_column_2 = rows_with_last_column_2(:,23:end-1);

rows_with_last_column_1 = temp_data(temp_data(:, end) == 1, :);
rows_with_last_column_1 = rows_with_last_column_1(:,23:end-1);

rows_with_last_column_3 = temp_data(temp_data(:, end) == 3, :);
rows_with_last_column_3 = rows_with_last_column_3(:,23:end-1);

% Apply abs to the variables
rows_with_last_column_1 = abs(rows_with_last_column_1);
rows_with_last_column_2 = abs(rows_with_last_column_2);
rows_with_last_column_3 = abs(rows_with_last_column_3);

% Initialize arrays to store means
mean_last_column_1 = zeros(1, size(rows_with_last_column_1, 2));
mean_last_column_2 = zeros(1, size(rows_with_last_column_2, 2));
mean_last_column_3 = zeros(1, size(rows_with_last_column_3, 2));

% Compute the mean for each column of rows_with_last_column_1
for col = 1:size(rows_with_last_column_1, 2)
    mean_last_column_1(col) = customMean(rows_with_last_column_1(:, col));
end

% Compute the mean for each column of rows_with_last_column_2
for col = 1:size(rows_with_last_column_2, 2)
    mean_last_column_2(col) = customMean(rows_with_last_column_2(:, col));
end

% Compute the mean for each column of rows_with_last_column_3
for col = 1:size(rows_with_last_column_3, 2)
    mean_last_column_3(col) = customMean(rows_with_last_column_3(:, col));
end

% Display the results
disp('Mean, Standard_Deviation, Entropy, RMS, Variance, Kurtosis, Skewness for class 1:');
disp(mean_last_column_1);

disp('Mean, Standard_Deviation, Entropy, RMS, Variance, Kurtosis, Skewness for class 2:');
disp(mean_last_column_2);

disp('Mean, Standard_Deviation, Entropy, RMS, Variance, Kurtosis, Skewness for class 3:');
disp(mean_last_column_3);
% Preparing Training Data SVM
y_train = train_data(:, end);
x_train = train_data(:, 1:end-1);

% Training SVM Model
svmTemplate = templateSVM('KernelFunction', 'rbf', ...
    'BoxConstraint', 7, 'KernelScale', 'auto', 'Standardize', true);
svmModel = fitcecoc(x_train, y_train, 'Learners', svmTemplate, 'ClassNames', unique(y_train));

disp('Multi-class SVM model trained successfully.');

% Preparing Test Data
y_test = test_data(:, end);
x_test = test_data(:, 1:end-1);

% Prediction Using Multi-Class SVM
predictedLabels_svm = predict(svmModel, x_test);

% Evaluating Model Performance
% Calculate Accuracy
accuracy_svm = sum(predictedLabels_svm == y_test) / length(y_test);
disp(['Accuracy SVM (%) : ', num2str(accuracy_svm * 100)]);

% Generating Confusion Matrix
confusionMat_svm = confusionmat(y_test, predictedLabels_svm);

% Calculating Class-wise Metrics (Precision, Recall, F1-Score)
numClasses = size(confusionMat_svm, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confusionMat_svm(i, i); % True Positives
    FP = sum(confusionMat_svm(:, i)) - TP; % False Positives
    FN = sum(confusionMat_svm(i, :)) - TP; % False Negatives

    precision(i) = TP / (TP + FP + eps); % Adding eps to avoid division by zero
    recall(i) = TP / (TP + FN + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Calculating Macro Metrics
macroPrecision_svm = mean(precision) * 100; % Average Precision
macroRecall_svm = mean(recall) * 100;  % Average Recall
macroF1Score_svm = mean(f1Score) * 100; % Average F1-Score

% Displaing Macro Metrics
disp(['SVM Macro Precision (%): ', num2str(macroPrecision_svm)]);
disp(['SVM Macro Recall (%): ', num2str(macroRecall_svm)]);
disp(['SVM Macro F1-Score (%): ', num2str(macroF1Score_svm)]);

% Plot Confusion Matrix
figure;
confChart = confusionchart(confusionMat_svm);
confChart.Title = 'Confusion Matrix for SVM';
confChart.XLabel = 'Predicted Labels';
confChart.YLabel = 'True Labels';

% Prediction scores for each class (needed for ROC-AUC)
[~, scores] = predict(svmModel, x_test); % 'scores' contains probabilities for each class

% Prepare variables for ROC
numClasses = size(scores, 2); % Number of classes
trueLabelsOneHot = zeros(size(y_test, 1), numClasses); % One-hot encode true labels
for i = 1:numClasses
    trueLabelsOneHot(:, i) = (y_test == i); % Create one-hot encoding for each class
end

% Plotting ROC Curve for each class
figure;
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores(:, i), 1); % Compute ROC curve
    plot(X, Y, 'LineWidth', 2,'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC)); % Plot ROC with AUC
end
hold off;

% Customize ROC Plot
title('ROC Curve for Multi-Class SVM');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show');
grid on;
% Set custom axis limits
axis([-0.1 1 -0.1 1.1]); % X-axis from -0.1 to 1, Y-axis from -0.1 to 1.1

% Preparing Training Data RF
y_train = train_data(:, end);
x_train = train_data(:, 1:end-1);

% Converting numeric labels to strings for TreeBagger
y_train_str = cellstr(num2str(y_train)); % Convert to cell array of strings

% Training Random Forest Classifier
numTrees = 200; % Number of trees in the forest
randomForestModel = TreeBagger(numTrees, x_train, y_train_str, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'ClassNames', cellstr(num2str(unique(y_train))));

disp('Random Forest model trained successfully.');

% Preparing Test Data
y_test = test_data(:, end);
x_test = test_data(:, 1:end-1);

% Prediction Using Random Forest
predictedLabels_str = predict(randomForestModel, x_test); % Predicted labels as strings
predictedLabels_rf = str2double(predictedLabels_str); % Convert back to numeric

% Evaluating Model Performance
% Calculate Accuracy
accuracy_rf = sum(predictedLabels_rf == y_test) / length(y_test);
disp(['Accuracy RF (%): ', num2str(accuracy_rf * 100)]);

% Generating Confusion Matrix
confusionMat_rf = confusionmat(y_test, predictedLabels_rf);

% Calculating Class-wise Metrics (Precision, Recall, F1-Score)
numClasses = size(confusionMat_rf, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confusionMat_rf(i, i); % True Positives
    FP = sum(confusionMat_rf(:, i)) - TP; % False Positives
    FN = sum(confusionMat_rf(i, :)) - TP; % False Negatives

    precision(i) = TP / (TP + FP + eps); % Add eps to avoid division by zero
    recall(i) = TP / (TP + FN + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Calculating Macro Metrics
macroPrecision_rf = mean(precision) * 100; % Average Precision
macroRecall_rf = mean(recall) * 100;       % Average Recall
macroF1Score_rf = mean(f1Score) * 100;    % Average F1-Score

% Displaying Macro Metrics
disp(['RF Macro Precision (%): ', num2str(macroPrecision_rf)]);
disp(['RF Macro Recall (%): ', num2str(macroRecall_rf)]);
disp(['RF Macro F1-Score (%): ', num2str(macroF1Score_rf)]);

% Plotting Confusion Matrix
figure;
confChart = confusionchart(confusionMat_rf);
confChart.Title = 'Confusion Matrix for Random Forest';
confChart.XLabel = 'Predicted Labels';
confChart.YLabel = 'True Labels';

% Predicting scores for each class (needed for ROC-AUC)
[~, scores_rf] = predict(randomForestModel, x_test); % 'scores_rf' contains probabilities for each class

% Preparing variables for ROC
trueLabelsOneHot = zeros(size(y_test, 1), numClasses); % One-hot encode true labels
for i = 1:numClasses
    trueLabelsOneHot(:, i) = (y_test == i); % Create one-hot encoding for each class
end

% Plot ROC Curve for each class
figure;
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_rf(:, i), 1); % Compute ROC curve
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC)); % Plot ROC with AUC
end
hold off;

% Customize ROC Plot
title('ROC Curve for Random Forest');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show');
grid on;
% Setting custom axis limits
axis([-0.1 1 -0.1 1.1]); % X-axis from -0.1 to 1, Y-axis from -0.1 to 1.1

% Training Gradient Boosting Classifier

y_train = train_data(:, end); % Target labels
x_train = train_data(:, 1:end-1); % Features

% Training Gradient Boosting Classifier
gbModel = fitcensemble(x_train, y_train, ...
    'Method', 'Bag', ...         % Use gradient boosting (Bagging/Boosting)
    'NumLearningCycles', 200, ... % Number of trees
    'Learners', templateTree('MaxNumSplits', 20)); % Weak learners with limited splits

disp('Gradient Boosting model trained successfully.');

% Preparing Test Data
y_test = test_data(:, end); % Target labels
x_test = test_data(:, 1:end-1); % Features

% Predicting Using Gradient Boosting
predictedLabels_gb = predict(gbModel, x_test);

% Evaluating Model Performance
% Calculate Accuracy
accuracy_gb = sum(predictedLabels_gb == y_test) / length(y_test);
disp(['Accuracy (Gradient Boosting): ', num2str(accuracy_gb * 100)]);

% Generating Confusion Matrix
confusionMat_gb = confusionmat(y_test, predictedLabels_gb);

% Calculating Class-wise Metrics (Precision, Recall, F1-Score)
numClasses = size(confusionMat_gb, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confusionMat_gb(i, i); % True Positives
    FP = sum(confusionMat_gb(:, i)) - TP; % False Positives
    FN = sum(confusionMat_gb(i, :)) - TP; % False Negatives

    precision(i) = TP / (TP + FP + eps); % Add eps to avoid division by zero
    recall(i) = TP / (TP + FN + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Calculating Macro Metrics
macroPrecision_gb = mean(precision) * 100; % Average Precision
macroRecall_gb = mean(recall) * 100;       % Average Recall
macroF1Score_gb = mean(f1Score) * 100;    % Average F1-Score

% Displaying Macro Metrics
disp(['GB Macro Precision (%): ', num2str(macroPrecision_gb)]);
disp(['GB Macro Recall (%): ', num2str(macroRecall_gb)]);
disp(['GB Macro F1-Score (%): ', num2str(macroF1Score_gb)]);

% Plot Confusion Matrix
figure;
confChart = confusionchart(confusionMat_gb);
confChart.Title = 'Confusion Matrix for Gradient Boosting';
confChart.XLabel = 'Predicted Labels';
confChart.YLabel = 'True Labels';

% Predicting scores for each class (needed for ROC-AUC)
[~, scores_gb] = predict(gbModel, x_test); % 'scores_gb' contains probabilities for each class

% Preparing variables for ROC
trueLabelsOneHot = zeros(size(y_test, 1), numClasses); % One-hot encode true labels
for i = 1:numClasses
    trueLabelsOneHot(:, i) = (y_test == i); % Create one-hot encoding for each class
end

% Ploting ROC Curve for each class
figure;
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_gb(:, i), 1); % Compute ROC curve
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC)); % Plot ROC with AUC
end
hold off;

% Customize ROC Plot
title('ROC Curve for Gradient Boosting');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show');
grid on;

% Setting custom axis limits
axis([-0.1 1 -0.1 1.1]); % X-axis from -0.1 to 1, Y-axis from -0.1 to 1.1

% Training KNN Classifier
% Preparing Training Data
y_train = train_data(:, end); % Target labels
x_train = train_data(:, 1:end-1); % Features

% Training KNN model
knnModel = fitcknn(x_train, y_train, ...
    'NumNeighbors', 5, ...        % Number of nearest neighbors
    'Standardize', true);         % Standardize features before training

disp('KNN model trained successfully.');

% Preparing Test Data
y_test = test_data(:, end); % Target labels
x_test = test_data(:, 1:end-1); % Features

% Predicting Using KNN
predictedLabels_knn = predict(knnModel, x_test);

% Evaluating Model Performance
% Calculate Accuracy
accuracy_knn = sum(predictedLabels_knn == y_test) / length(y_test);
disp(['Accuracy (KNN): ', num2str(accuracy_knn * 100)]);

% Generating Confusion Matrix
confusionMat_knn = confusionmat(y_test, predictedLabels_knn);

% Calculate Class-wise Metrics (Precision, Recall, F1-Score)
numClasses = size(confusionMat_knn, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confusionMat_knn(i, i); % True Positives
    FP = sum(confusionMat_knn(:, i)) - TP; % False Positives
    FN = sum(confusionMat_knn(i, :)) - TP; % False Negatives

    precision(i) = TP / (TP + FP + eps); % Add eps to avoid division by zero
    recall(i) = TP / (TP + FN + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Calculating Macro Metrics
macroPrecision_knn = mean(precision) * 100; % Average Precision
macroRecall_knn = mean(recall) * 100;       % Average Recall
macroF1Score_knn = mean(f1Score) * 100;    % Average F1-Score

% Displaing Macro Metrics
disp(['KNN Macro Precision (%): ', num2str(macroPrecision_knn)]);
disp(['KNN Macro Recall (%): ', num2str(macroRecall_knn)]);
disp(['KNN Macro F1-Score (%): ', num2str(macroF1Score_knn)]);

% Plotting Confusion Matrix
figure;
confChart_knn = confusionchart(confusionMat_knn);
confChart_knn.Title = 'Confusion Matrix for KNN';
confChart_knn.XLabel = 'Predicted Labels';
confChart_knn.YLabel = 'True Labels';

% Predicting scores for each class (needed for ROC-AUC)
[~, scores_knn] = predict(knnModel, x_test); % 'scores_knn' contains probabilities for each class

% Preparing variables for ROC
trueLabelsOneHot = zeros(size(y_test, 1), numClasses); % One-hot encode true labels
for i = 1:numClasses
    trueLabelsOneHot(:, i) = (y_test == i); % Create one-hot encoding for each class
end

% Plotting ROC Curve for each class
figure;
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_knn(:, i), 1); % Compute ROC curve
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC)); % Plot ROC with AUC
end
hold off;

% Customize ROC Plot
title('ROC Curve for KNN');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show');
grid on;

% Setting custom axis limits
axis([-0.1 1 -0.1 1.1]); % X-axis from -0.1 to 1, Y-axis from -0.1 to 1.1

% Radar plot
% Defining metrics of each model
metrics_knn = [accuracy_knn * 100, macroPrecision_knn, macroRecall_knn, macroF1Score_knn];
metrics_svm = [accuracy_svm * 100, macroPrecision_svm, macroRecall_svm, macroF1Score_svm];
metrics_rf = [accuracy_rf * 100, macroPrecision_rf, macroRecall_rf, macroF1Score_rf];
metrics_gb = [accuracy_gb * 100, macroPrecision_gb, macroRecall_gb, macroF1Score_gb];

% Combining all metrics into a matrix
all_metrics = [metrics_knn; metrics_svm; metrics_rf; metrics_gb];

% Labels for the axes
metric_labels = {'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'};

% Number of metrics
numMetrics = size(all_metrics, 2);

% Angles for the radar chart (slightly adjust each model's angle to avoid overlap)
angles = linspace(0, 2*pi, numMetrics + 1); % Add extra angle to close the chart
angle_offsets = [-0.1, -0.05, 0.05, 0.1]; % Offset for each model to create separation

% Extend the data for circular plotting
all_metrics = [all_metrics, all_metrics(:, 1)];

% Define custom colors for each model
custom_colors = [0, 0.4470, 0.7410; % Blue for KNN
                 0.8500, 0.3250, 0.0980; % Orange for SVM
                 0.9290, 0.6940, 0.1250; % Yellow for RF
                 0.4940, 0.1840, 0.5560]; % Purple for GB

% Creating figure and polar axes
figure;
p = polaraxes;
hold(p, 'on');

% Plotting points on radar chart for each model with offsets
for i = 1:size(all_metrics, 1)
    adjusted_angles = angles + angle_offsets(i); % Apply offset for the model
    polarplot(p, adjusted_angles, all_metrics(i, :), 'p', ...
        'MarkerSize', 16, 'Color', custom_colors(i, :), 'MarkerFaceColor', custom_colors(i, :));
end

% Customize plot
p.ThetaTick = rad2deg(angles(1:end-1)); % Set axis ticks
p.ThetaTickLabel = metric_labels;       % Set axis labels
p.RLim = [0 100];                       % Set radial axis limits
p.LineWidth = 1.5;                      % Set grid line width

% Adding legend
legend({'KNN', 'SVM', 'RF', 'GB'}, 'Location', 'southoutside', 'Orientation', 'horizontal');

% Adding title
title('Radar Chart: Comparison of Classification Models');

hold(p, 'off');

% Bar Chart
% Define metrics for each model
metrics_knn = [accuracy_knn * 100, macroPrecision_knn, macroRecall_knn, macroF1Score_knn];
metrics_svm = [accuracy_svm * 100, macroPrecision_svm, macroRecall_svm, macroF1Score_svm];
metrics_rf = [accuracy_rf * 100, macroPrecision_rf, macroRecall_rf, macroF1Score_rf];
metrics_gb = [accuracy_gb * 100, macroPrecision_gb, macroRecall_gb, macroF1Score_gb];

% Combine all metrics into a matrix
all_metrics = [metrics_knn; metrics_svm; metrics_rf; metrics_gb];

% Labels for the axes
metric_labels = {'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'};
model_labels = {'KNN', 'SVM', 'RF', 'GB'};

% Create figure
figure;
bar_chart = bar(all_metrics', 'grouped'); % Transpose for grouped bar chart

% Customize bar colors
custom_colors = [0, 0.4470, 0.7410; % Blue for KNN
                 0.8500, 0.3250, 0.0980; % Orange for SVM
                 0.9290, 0.6940, 0.1250; % Yellow for RF
                 0.4940, 0.1840, 0.5560]; % Purple for GB

for i = 1:size(all_metrics, 1)
    bar_chart(i).FaceColor = 'flat';
    bar_chart(i).CData = repmat(custom_colors(i, :), size(all_metrics, 2), 1);
end

% Add labels, legend, and title
xticks(1:numel(metric_labels));
xticklabels(metric_labels); % Set x-axis labels
xlabel('Metrics');
ylabel('Percentage (%)');
title('Bar Chart: Comparison of Classification Models');
legend(model_labels, 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;

% Adjust figure properties
set(gcf, 'Color', 'w'); % Set background color to white
%%
% Subplot for Confusion Matrices
figure('Name', 'Confusion Matrices', 'NumberTitle', 'off');

% SVM Confusion Matrix
subplot(2, 2, 1);
confChart_svm = confusionchart(confusionMat_svm);
confChart_svm.Title = 'Confusion Matrix: SVM';
confChart_svm.XLabel = 'Predicted Labels';
confChart_svm.YLabel = 'True Labels';

% Random Forest Confusion Matrix
subplot(2, 2, 2);
confChart_rf = confusionchart(confusionMat_rf);
confChart_rf.Title = 'Confusion Matrix: Random Forest';
confChart_rf.XLabel = 'Predicted Labels';
confChart_rf.YLabel = 'True Labels';

% Gradient Boosting Confusion Matrix
subplot(2, 2, 3);
confChart_gb = confusionchart(confusionMat_gb);
confChart_gb.Title = 'Confusion Matrix: Gradient Boosting';
confChart_gb.XLabel = 'Predicted Labels';
confChart_gb.YLabel = 'True Labels';

% KNN Confusion Matrix
subplot(2, 2, 4);
confChart_knn = confusionchart(confusionMat_knn);
confChart_knn.Title = 'Confusion Matrix: KNN';
confChart_knn.XLabel = 'Predicted Labels';
confChart_knn.YLabel = 'True Labels';

% Subplot for ROC AUC Curves
figure('Name', 'ROC AUC Curves', 'NumberTitle', 'off');

% SVM ROC AUC
subplot(2, 2, 1);
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores(:, i), 1);
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC));
end
hold off;
title('ROC Curve: SVM');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'southeast');
grid on;
axis([-0.1 1 -0.1 1.1]);

% Random Forest ROC AUC
subplot(2, 2, 2);
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_rf(:, i), 1);
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC));
end
hold off;
title('ROC Curve: Random Forest');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'southeast');
grid on;
axis([-0.1 1 -0.1 1.1]);

% Gradient Boosting ROC AUC
subplot(2, 2, 3);
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_gb(:, i), 1);
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC));
end
hold off;
title('ROC Curve: Gradient Boosting');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'southeast');
grid on;
axis([-0.1 1 -0.1 1.1]);

% KNN ROC AUC
subplot(2, 2, 4);
hold on;
for i = 1:numClasses
    [X, Y, T, AUC] = perfcurve(trueLabelsOneHot(:, i), scores_knn(:, i), 1);
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %d (AUC = %.2f)', i, AUC));
end
hold off;
title('ROC Curve: KNN');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'southeast');
grid on;
axis([-0.1 1 -0.1 1.1]);
