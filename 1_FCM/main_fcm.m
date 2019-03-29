clear;
clc;

% 对输入图像进行各向异性扩散滤波
% 然后利用FCM分割

path = 'F:\研究僧科研\7毕业论文实验\image\test\';
img_list = dir([path, '*.png']);
img_num = length(img_list);
num = 7;

% for num = 1:img_num
    img_name = img_list(num).name;
    I = imread(strcat(path,img_name));
%     figure;
%     subplot(121);
%     imshow(I);
    [m, n, dim] = size(I);
    if dim>1
        I_ori = rgb2gray(I);
    end
%     J(:,:,1) = I;
%     J(:,:,2) = I;
%     J(:,:,3) = I;
    I = double(I_ori);          % I：原图
    
    %% ***************各向异性扩散滤波*****************
    niter = 20;
    lambda = 0.1;
    kappa = 20;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM：各向异性扩散滤波后图像
%     figure;imshow(uint8(I_PM));title('PM20_01');

    %% ***************FCM分割*****************
    data = reshape(I_PM, numel(I_PM), 1);
    cluster_n = 5;
    [center, U, obj_fcn] = fcm(data, cluster_n);
    maxU = max(U);
    temp = sort(center);
    for i=1:cluster_n
        eval(['cluster',int2str(i),'_index = find(U(',int2str(i),',:) == maxU);']);
        index = find(temp == center(i));
        switch index
            case 1
                color_class = 0;
            case cluster_n 
                color_class = 255;
            otherwise
                color_class = fix(255*(index-1)/(cluster_n-1));
        end
        eval(['I(cluster',int2str(i),'_index) =',int2str(color_class),';']);
    end
    I = mat2gray(I);
    figure;imshow(I)
% end