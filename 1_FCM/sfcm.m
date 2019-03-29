function [center, U, obj_fcn] =sfcm(img,c,options) 
% 输入：
%   img 大小为m行n列的图像
%   c 聚类中心的个数
%   options(1): 空间系数alpha，邻域的参考程度(缺省值: 0.1)
%   options(2): 高斯函数方差的sigma(缺省值: 1)
%   options(3): N邻域大小(缺省值: 3,代表3*3)
%   options(4): 隶属度矩阵U的指数expo，>1(缺省值: 2.0)
%   options(5): 最大迭代次数max_t(缺省值: 100)
%   options(6): 隶属度最小变化量e,迭代终止条件(缺省值: 1e-5)
%   options(7): 每次迭代是否输出信息标志(缺省值: 1)
% 输出：
%   U 隶属度矩阵
%   center 聚类中心
%   obj_fcn 目标函数

% 判断输入参数的个数只能是2个或者3个
if nargin ~= 2 && nargin ~= 3
    error('Too many or too few input argument! ');
end
data_n =numel(img);     % 图像大小,像素点即样本个数
% 默认操作参数
default_options = [0.25; 250; 5; 2; 100; 1e-5; 1];

% 如果输入参数个数为2，调用默认的options参数
if nargin == 2,
    options = default_options;
else % 分析有options做参数的情况：
     % 用户在输入options参数时需注意，如果options参数的个数少于6个，则未输入参数的对应位置用nan来代替，这样可以保证未输入参数采用的是默认值，否则可能会出现前面参数占用后面参数的值的情况，从而影响聚类效果。
    if length(options) < 7,
        temp = default_options;
        temp(1:length(options)) = options;
        options = temp;
    end
    % 返回options中值为NaN的索引位置
    nan_index = find(isnan(options) == 1);
    % 将default_options中对应位置的参数值付给options中值为NaN的值
    options(nan_index) = default_options(nan_index);
    if options(3)<=1 ||options(4) <= 1,% 模糊因子是大于1的数
        error('The exponent or neiborhold should be greater than 1 !');
    end

end

% 将options中的分量分别复制给四个变量
alpha = options(1); %空间系数alpha
sigma = options(2); %方差sigma
N = int32(options(3));     %邻域
expo = options(4);  %模糊因子m
max_t = options(5); % 最大迭代次数
e = options(6);     %迭代终止条件
display = options(7);%输出信息


% 目标函数初始化
obj_fcn = zeros(max_t, 1);
U = initfcm(c, data_n);
% 初始化模糊分配矩阵，使U满足列上相加值为1

% 主要循环
for i = 1 : max_t
    % 在第k步循环中改变聚类中心center和隶属矩阵U
    [U, center, obj_fcn(i)] = stepfcm(img, U, c, expo, alpha, sigma, N);
    if display,
        fprintf('SFCM:Iteration count = %d, obj_fcn = %f\n',i,obj_fcn(i));
    end

    %迭代终止条件的判断
    if i > 1,
        if abs(obj_fcn(i) - obj_fcn(i-1)) < e,
            break;
        end
    end

end

% 实际迭代次数
iter_n = i;
% 清除迭代次数之后的值
obj_fcn(iter_n + 1 : max_t) = [];
end