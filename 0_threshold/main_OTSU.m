clear;
clc;

% 对输入图像进行各向异性扩散滤波
% 然后整张图片利用OTSU进行阈值分割

path = 'F:\研究僧科研\7毕业论文实验\image\train\';
img_list = dir([path, '*.png']);
img_num = length(img_list);
num = 17;
% for num = 1:img_num
    img_name = img_list(num).name;
    I = imread(strcat(path,img_name));
%     figure;
%     subplot(121);
%     imshow(I);
    [m, n, dim] = size(I);
    if dim>1
        I = rgb2gray(I);
    end
%     J(:,:,1) = I;
%     J(:,:,2) = I;
%     J(:,:,3) = I;
    I = double(I);          % I：原图
%     
    %% ***************各向异性扩散滤波*****************
    niter = 60;
    lambda = 0.1;
    kappa = 60;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM：各向异性扩散滤波后图像
%     figure;imshow(uint8(I_PM));title('PM20_01');
    I_bw = uint8(I);
    %% OTSU分割
    level = graythresh(I_bw);     %确定灰度阈值
    BW = im2bw(I_bw,level);
%     subplot(122);
    figure;
    imshow(BW);
% end