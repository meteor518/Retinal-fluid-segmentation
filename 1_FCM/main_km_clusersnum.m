clear;
clc;

% ������ͼ����и���������ɢ�˲�
% ����kmeans�㷨�����ƾ��������ͼ��ȷ���������ĸ���

path = 'F:\�о�ɮ����\7��ҵ����ʵ��\image\train_retina\';
img_list = dir([path, '*.png']);
img_num = length(img_list);
num = 6;

img_name = img_list(num).name;
I = imread(strcat(path,img_name));
[m, n, dim] = size(I);
if dim>1
    I = rgb2gray(I);
end
I = double(I);          % I��ԭͼ
I = imresize(I, [m/2,n/2]);

%% ***************����������ɢ�˲�*****************
niter = 20;
lambda = 0.1;
kappa = 20;
option = 1;
I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM������������ɢ�˲���ͼ��
%     figure;imshow(uint8(I_PM));title('PM20_01');

%% ***************Kmeans��������ͼȷ���������ĸ���*****************
data = reshape(I_PM, numel(I_PM), 1);
cluster_n  = 4;
id = kmeans(data, cluster_n);
figure;[silh, h] = silhouette(data, id);
xlabel('silhouetteֵ'); ylabel('����������')