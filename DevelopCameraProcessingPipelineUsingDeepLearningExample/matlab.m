%cfa = imread('/Users/user/Downloads/ZurichRAWtoRGB/test/huawei_raw/45.png'); % Read your grayscale image
 %cfa = imread('/Users/user/Downloads/ZurichRAWtoRGB/test/huawei_full_resolution/2169.png'); % Read your grayscale image
%cfa = imread('/Users/user/Downloads/ZurichRAWtoRGB/train/huawei_raw/103.png'); % Read your grayscale image
%cfa = imread('/Users/user/Downloads/ZurichRAWtoRGB/test/huawei_raw/17.png'); % Read your grayscale image
cfa = imread('/Users/user/Downloads/ZurichRAWtoRGB/test/huawei_raw/9.png'); % Read your grayscale image
cfa2 = imread('/Users/user/Downloads/ZurichRAWtoRGB/test/huawei_visualized/9.jpg');
% Extract Bayer pattern channels
blue = cfa(2:2:end, 2:2:end);
green1 = cfa(1:2:end, 2:2:end);
red = cfa(1:2:end, 1:2:end);
green2 = cfa(2:2:end, 1:2:end);

% Combine Bayer pattern channels
rawCombined = cat(3, blue, green1, red, green2);

% Normalize to [0, 1] (assuming 10-bit sensor)
out = single(rawCombined) / (4 * 255);

% Display or further process the Bayer pattern image
% imshow(out);


saveVarsMat = load('matlab.mat');

net = saveVarsMat.net; % <1x1 dlnetwork> unsupported class
% 
% disp(net)
% 
% summary(net)

networkInput = dlarray((out),"SSC");
output = forward(net,networkInput)


image_data = extractdata(output);

imshow(image_data);

% cfa2_resized = imresize(cfa2, [448 448]);

% Convert cfa2 to single
% cfa2_single = im2single(cfa2_resized);
% 
% scores = multissim(image_data,cfa2_single);
% 
% disp(squeeze(scores));

% imwrite(image_data, file_path);