% ���룺data, center
% �����out

function out = disfcm(center,data)
out = zeros(size(center, 1), size(data, 1)); % ��c��n�еľ�����������������
% ѭ����ÿѭ��һ�μ������������㵽�þ������ĵľ���
for k = 1:size(center, 1)
    out(k,:) = sqrt(sum(((data - ones(size(data,1),1)*center(k,:)).^2)',1));
end
end