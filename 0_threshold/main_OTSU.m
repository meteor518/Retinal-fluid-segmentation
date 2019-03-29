clear;
clc;

% ������ͼ����и���������ɢ�˲�
% Ȼ������ͼƬ����OTSU������ֵ�ָ�

path = 'F:\�о�ɮ����\7��ҵ����ʵ��\image\train\';
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
    I = double(I);          % I��ԭͼ
%     
    %% ***************����������ɢ�˲�*****************
    niter = 60;
    lambda = 0.1;
    kappa = 60;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM������������ɢ�˲���ͼ��
%     figure;imshow(uint8(I_PM));title('PM20_01');
    I_bw = uint8(I);
    %% OTSU�ָ�
    level = graythresh(I_bw);     %ȷ���Ҷ���ֵ
    BW = im2bw(I_bw,level);
%     subplot(122);
    figure;
    imshow(BW);
% end