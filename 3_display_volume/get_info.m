function [bubbleline,LTnum] = get_info(I_bw)
% ����Ŀ�꣺�õ�ˮ�׵�λ����Ϣ
% I_bw: ˮ�׷ָ��Ķ�ֵͼ
% LTnum: ˮ�׵ĸ���
% bubbleline: LTnum*3�����ڴ洢ˮ���������Ϣ����һ�д�ˮ�����ĸ߶ȣ������д�ˮ�����ұ߽�

%% ����I_bwͼ����ж�ˮ�ݱ��
[LT,LTnum] = bwlabel(I_bw);
S = regionprops(LT,'all');

%% �ҳ�ˮ��ͶӰ�����ͼ�Ŀ��
bubbleline = zeros(LTnum,3); % bubbleline���ڴ�ˮ��ӳ����Ϣ

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