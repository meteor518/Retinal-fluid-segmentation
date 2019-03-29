% һ�ξ�������Ĺ��̣�
% % ��1������������ģ���2��Ŀ�꺯������3�����뺯������4�������µ���������

% ���룺data,U,c,expo(ģ������),alpha(�ռ�ϵ��),sigma(��˹����),N(����Χ)
% �����U_new���µ��������󣩣�center���������ģ���obj_fcn��Ŀ�꺯����

function[U_new, center, obj_fcn] = stepfcm(img, U, c, expo, alpha, sigma, N)

% �������������ָ�����㣨����ģ�����ӣ�
mf = U.^expo; % ��������ģ����
% ����������������������ֵ
[m, n] = size(img);
pad = (N-1)/2;
img_pad = zeros(m+2*pad, n+2*pad);
new_img = img_pad;
img_pad(pad+1:m+pad, pad+1:n+pad) = img;

%**********��Ȩ��WLab**********
for i=pad+1:m+pad
    for j=pad+1:n+pad 
        temp = ones(N, N)* img_pad(i, j); % ����������ƽ�̵���������
        w = exp(-(new_img(i-pad:i+pad, j-pad:j+pad)-temp)^2/(2.0*sigma^2));% ���ø�˹������������Ȩ��
        x_ = sum(sum(w.*img_pad(i-pad:i+pad, j-pad:j+pad))) / sum(sum(w)); % ���������ƽ��ֵ
        new_img(i, j) = (img_pad(i, j) + alpha*x_) / (1+alpha);    % ԭ����������ֵ
    end
end
new_img = new_img(pad+1:m+pad, pad+1:n+pad);
data = reshape(new_img, numel(new_img), 1);
% �൱��һ���Խ����о���������Ϊһ������������
center = mf*data./((ones(size(data,2),1)*sum(mf'))');
dist = distfcm(center, data); % ����������
obj_fcn = sum(sum(dist.^2 .* mf)); % ����Ŀ�꺯��
temp = dist.^(-2/(expo-1));
U_new = temp./(ones(c,1)*sum(temp)); % �����µ���������
end