% 输入：data, center
% 输出：out

function out = disfcm(center,data)
out = zeros(size(center, 1), size(data, 1)); % 对c行n列的距离输出矩阵进行置零
% 循环，每循环一次计算所有样本点到该聚类中心的距离
for k = 1:size(center, 1)
    out(k,:) = sqrt(sum(((data - ones(size(data,1),1)*center(k,:)).^2)',1));
end
end