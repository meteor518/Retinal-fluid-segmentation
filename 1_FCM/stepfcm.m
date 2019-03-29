% 一次聚类包含的过程：
% % （1）计算聚类中心，（2）目标函数，（3）距离函数，（4）计算新的隶属矩阵

% 输入：data,U,c,expo(模糊因子),alpha(空间系数),sigma(高斯方差),N(邻域范围)
% 输出：U_new（新的隶属矩阵），center（聚类中心），obj_fcn（目标函数）

function[U_new, center, obj_fcn] = stepfcm(img, U, c, expo, alpha, sigma, N)

% 对隶属矩阵进行指数运算（加上模糊因子）
mf = U.^expo; % 隶属矩阵模糊化
% 计算样本考虑邻域后的修正值
[m, n] = size(img);
pad = (N-1)/2;
img_pad = zeros(m+2*pad, n+2*pad);
new_img = img_pad;
img_pad(pad+1:m+pad, pad+1:n+pad) = img;

%**********求权重WLab**********
for i=pad+1:m+pad
    for j=pad+1:n+pad 
        temp = ones(N, N)* img_pad(i, j); % 将中心像素平铺到整个邻域
        w = exp(-(new_img(i-pad:i+pad, j-pad:j+pad)-temp)^2/(2.0*sigma^2));% 利用高斯函数计算邻域权重
        x_ = sum(sum(w.*img_pad(i-pad:i+pad, j-pad:j+pad))) / sum(sum(w)); % 邻域的样本平均值
        new_img(i, j) = (img_pad(i, j) + alpha*x_) / (1+alpha);    % 原样本的修正值
    end
end
new_img = new_img(pad+1:m+pad, pad+1:n+pad);
data = reshape(new_img, numel(new_img), 1);
% 相当于一次性将所有聚类中心作为一个矩阵计算出来
center = mf*data./((ones(size(data,2),1)*sum(mf'))');
dist = distfcm(center, data); % 计算距离矩阵
obj_fcn = sum(sum(dist.^2 .* mf)); % 计算目标函数
temp = dist.^(-2/(expo-1));
U_new = temp./(ones(c,1)*sum(temp)); % 计算新的隶属矩阵
end