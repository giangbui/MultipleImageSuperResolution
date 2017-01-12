folder = '91';
outDir = 'Train';
imgfiles = dir(folder);
fid = fopen(fullfile(outDir,'img_list.txt'),'w');
for i=3:length(imgfiles)
    img = imread(fullfile(folder,imgfiles(i).name));
    [H,W] = size(img(:,:,1));
    Z = max(H,W);
    img2 = zeros(Z,Z,3,'uint8');
    img2(1:H,1:W,:) = img;
    if(Z>128)
        img2 = imresize(img2,[128,128]);
        imwrite(img2,fullfile(outDir,imgfiles(i).name));
        fprintf(fid,'%s\n',fullfile(outDir,imgfiles(i).name));
    end
end
fclose(fid);