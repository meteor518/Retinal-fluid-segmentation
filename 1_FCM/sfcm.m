function [center, U, obj_fcn] =sfcm(img,c,options) 
% ���룺
%   img ��СΪm��n�е�ͼ��
%   c �������ĵĸ���
%   options(1): �ռ�ϵ��alpha������Ĳο��̶�(ȱʡֵ: 0.1)
%   options(2): ��˹���������sigma(ȱʡֵ: 1)
%   options(3): N�����С(ȱʡֵ: 3,����3*3)
%   options(4): �����Ⱦ���U��ָ��expo��>1(ȱʡֵ: 2.0)
%   options(5): ����������max_t(ȱʡֵ: 100)
%   options(6): ��������С�仯��e,������ֹ����(ȱʡֵ: 1e-5)
%   options(7): ÿ�ε����Ƿ������Ϣ��־(ȱʡֵ: 1)
% �����
%   U �����Ⱦ���
%   center ��������
%   obj_fcn Ŀ�꺯��

% �ж���������ĸ���ֻ����2������3��
if nargin ~= 2 && nargin ~= 3
    error('Too many or too few input argument! ');
end
data_n =numel(img);     % ͼ���С,���ص㼴��������
% Ĭ�ϲ�������
default_options = [0.25; 250; 5; 2; 100; 1e-5; 1];

% ��������������Ϊ2������Ĭ�ϵ�options����
if nargin == 2,
    options = default_options;
else % ������options�������������
     % �û�������options����ʱ��ע�⣬���options�����ĸ�������6������δ��������Ķ�Ӧλ����nan�����棬�������Ա�֤δ����������õ���Ĭ��ֵ��������ܻ����ǰ�����ռ�ú��������ֵ��������Ӷ�Ӱ�����Ч����
    if length(options) < 7,
        temp = default_options;
        temp(1:length(options)) = options;
        options = temp;
    end
    % ����options��ֵΪNaN������λ��
    nan_index = find(isnan(options) == 1);
    % ��default_options�ж�Ӧλ�õĲ���ֵ����options��ֵΪNaN��ֵ
    options(nan_index) = default_options(nan_index);
    if options(3)<=1 ||options(4) <= 1,% ģ�������Ǵ���1����
        error('The exponent or neiborhold should be greater than 1 !');
    end

end

% ��options�еķ����ֱ��Ƹ��ĸ�����
alpha = options(1); %�ռ�ϵ��alpha
sigma = options(2); %����sigma
N = int32(options(3));     %����
expo = options(4);  %ģ������m
max_t = options(5); % ����������
e = options(6);     %������ֹ����
display = options(7);%�����Ϣ


% Ŀ�꺯����ʼ��
obj_fcn = zeros(max_t, 1);
U = initfcm(c, data_n);
% ��ʼ��ģ���������ʹU�����������ֵΪ1

% ��Ҫѭ��
for i = 1 : max_t
    % �ڵ�k��ѭ���иı��������center����������U
    [U, center, obj_fcn(i)] = stepfcm(img, U, c, expo, alpha, sigma, N);
    if display,
        fprintf('SFCM:Iteration count = %d, obj_fcn = %f\n',i,obj_fcn(i));
    end

    %������ֹ�������ж�
    if i > 1,
        if abs(obj_fcn(i) - obj_fcn(i-1)) < e,
            break;
        end
    end

end

% ʵ�ʵ�������
iter_n = i;
% �����������֮���ֵ
obj_fcn(iter_n + 1 : max_t) = [];
end