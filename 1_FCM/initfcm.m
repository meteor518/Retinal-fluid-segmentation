% U--> c * n(c行n列) ，c为聚类数, n为样本数
% 初始化fcm的隶属度函数矩阵，满足列向量之和为1
%% 初始化隶属矩阵时需要已知矩阵的行和列，因此输入应该为该矩阵的行和列

% 输入: c, data_n
% data_n  数据集data所含的样本数
% c       这组数据的聚类数
% 输出：U（初始化之后的隶属矩阵）

function[U] = initfcm(c, data_n)

% 初始化隶属矩阵U：rand函数可产生在(0, 1)之间均匀分布的随机数组成的数组。
U = rand(c, data_n);    

%% 注意：隶属矩阵在初始化时需满足某数据j对各个聚类中心i(1<i<c)的隶属度之和为1.
%%% sum(U)是指对矩阵U进行纵向相加，即求每一列的和，结果是一个1行n列的矩阵。(sum函数后面参数不指定或指定为1时均表示列相加)。

col_sum = sum(U);

U = U./col_sum(ones(c,1), : );

% col_sum(ones(cluster_n, 1),:)等效于ones(clusters_n,1)*col_sum
% 上述目的是将col_sum扩展成与U（c,data_n）大小相同的矩阵，然后进行对应元素的点除，使隶属矩阵列项和为1。

%% 这里必须要多说一句，因为我一开始死活不理解这一句怎么能使得隶属矩阵列项和为1。建议可以用一组简单的数据做尝试，之后你会发现，它就是一道简单的数学题！！

end