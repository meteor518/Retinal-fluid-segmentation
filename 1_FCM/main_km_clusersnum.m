clear;
clc;

% 对输入图像进行各向异性扩散滤波
% 根据kmeans算法，绘制聚类的轮廓图，确定聚类中心个数

path = 'F:\研究僧科研\7毕业论文实验\image\train_retina\';
img_list = dir([path, '*.png']);
img_num = length(img_list);
num = 6;

img_name = img_list(num).name;
I = imread(strcat(path,img_name));
[m, n, dim] = size(I);
if dim>1
    I = rgb2gray(I);
end
I = double(I);          % I：原图
I = imresize(I, [m/2,n/2]);

%% ***************各向异性扩散滤波*****************
niter = 20;
lambda = 0.1;
kappa = 20;
option = 1;
I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM：各向异性扩散滤波后图像
%     figure;imshow(uint8(I_PM));title('PM20_01');

%% ***************Kmeans根据轮廓图确定聚类中心个数*****************
data = reshape(I_PM, numel(I_PM), 1);
cluster_n  = 4;
id = kmeans(data, cluster_n);
figure;[silh, h] = silhouette(data, id);
xlabel('silhouette值'); ylabel('聚类的类别数')