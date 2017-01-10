clear;close all;
%% settings
folder = 'Test/Set5';
output = '4Test/Set5';

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image)    
    image = imresize(imresize(image,1/3.0,'bicubic'),size(image(:,:,1)),'bicubic');
    
    imwrite(image,fullfile(output,filepaths(i).name));
       

end
