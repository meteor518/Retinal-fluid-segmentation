function [bubbleline,LTnum] = get_info(I_bw)
% 函数目标：得到水肿的位置信息
% I_bw: 水肿分割后的二值图
% LTnum: 水肿的个数
% bubbleline: LTnum*3，用于存储水肿自身的信息，第一列存水泡中心高度，二三列存水泡左右边界

%% 基于I_bw图像进行对水泡标号
[LT,LTnum] = bwlabel(I_bw);
S = regionprops(LT,'all');

%% 找出水泡投影到左侧图的宽度
bubbleline = zeros(LTnum,3); % bubbleline用于存水泡映射信息

for k=1:LTnum
    bubbleline(k,1) = floor(S(k).Centroid(2));
    bubbleline(k,2) = floor(S(k).BoundingBox(1));
    bubbleline(k,3) = floor(S(k).BoundingBox(1)+S(k).BoundingBox(3));

    for j=floor(S(k).BoundingBox(1)):floor(S(k).BoundingBox(1)+S(k).BoundingBox(3))
        for i=floor(S(k).BoundingBox(2)+S(k).BoundingBox(4)):-1:floor(S(k).BoundingBox(2))
            if LT(i,j)==k
                break
            end
        end         
    end
end