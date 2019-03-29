clear;
clc;

% ������ͼ����и���������ɢ�˲�
% Ȼ�����ûҶȺ��ݶȱ仯����ILM��RPE��ָ�
% �����ֵ�ָ�

path = 'F:\�о�ɮ����\7��ҵ����ʵ��\image\test\';
img_list = dir([path, '*.png']);
img_num = length(img_list);
num = 8;

% for num = 1:img_num
    img_name = img_list(num).name;
    I = imread(strcat(path,img_name));
%     figure;
%     subplot(121);
%     imshow(I);
    [m, n, dim] = size(I);
    if dim>1
        I = rgb2gray(I);
    end
    J(:,:,1) = I;
    J(:,:,2) = I;
    J(:,:,3) = I;
    I = double(I);          % I��ԭͼ
    
    %% ***************����������ɢ�˲�*****************
    niter = 60;
    lambda = 0.1;
    kappa = 60;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM������������ɢ�˲���ͼ��
%     figure;imshow(uint8(I_PM));title('PM20_01');

    %% ****************�ָ�ILM��RPE*******************
    gray_deback = 180;      % gray_deback:���ӽ���Ŀ��ĻҶ�ֵ������ȥ����������
    gray_back = 255;        % gray_back:�����Ҷ���ֵ
    gray_obj = 190;         % gray_obj��Ŀ�����ȷ��ֵ��

    %% ��һ�ζ�ֵ��
    % �ӽ��ڻ�ɫֵ�ļ�ΪĿ��������Ϊ��ɫ������ɫΪ��ɫ
    % ��ֵ����Ϊ�˵õ�ILM��RPE���ȥ����������
    I_bw = I_PM;
    black = abs(I_bw-gray_deback*ones(m,n));
    white = abs(I_bw-gray_back*ones(m,n));
    I_bw = (black < white) * 255;
    % ȥ�����С��200��������
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
%     figure;imshow(LT);title('��ֵ��');hold on
%     for i=1:LTnum
%         rectangle('position',S(i).BoundingBox,'edgecolor','r');
%     end 
    I_bw = ismember(LT, find([S.Area]>200));
%     figure;imshow(I_bw);title('��ֵ��');
   
    %% ������ϲ�ILM��RPE
    [ILMrow, ILMcol, ILMnum, RPErow, RPEcol, RPEnum] = ILMRPE_seg(I_bw, I_PM, gray_deback);
    
    for i=1:ILMnum
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),1) = 255;  % ��ILM����Ϊ��ɫ
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),2) = 0;
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),3) = 0;
    end   
%     figure;imshow(uint8(J));title('ILM');
    
    for i=1:RPEnum
        J(RPErow(i):RPErow(i)+2,RPEcol(i),1) = 255;  % ��RPE����Ϊ��ɫ
        J(RPErow(i):RPErow(i)+2,RPEcol(i),2) = 255;
        J(RPErow(i):RPErow(i)+2,RPEcol(i),3) = 0;
    end
%     figure; imshow(J)
    
    %% ****************��ֵ�ָ�ˮ��*******************
    % ��I_PMͼ��ȥ�����������������ұ�Ϊ��ɫ
    %RPE�����µı�����Ϊ��ɫ
    for j=1:RPEnum
        I_PM(RPErow(j):m,RPEcol(j)) = 0;  
    end
    %ILM�����ϵı�����Ϊ��ɫ
    for j=1:RPEnum
        I_PM(1:ILMrow(j),RPEcol(j)) = 0;  
    end
    % ���ҵ�Ϊ�磬����Ĥ���ҵı�����Ϊ��ɫ
    for j=RPEcol(RPEnum)+1:j
        I_PM(1:m,j) = 0;   
    end
    % �����Ϊ�磬����Ĥ���ҵı�����Ϊ��ɫ
    for j=1:RPEcol(1)-1
        I_PM(1:m,j) = 0;   
    end
%     figure;imshow(uint8(I_PM));title('ȥ�������I');
   
    % ���ζ�ֵ����׼ȷ�ָ�Ŀ������ˮ��Ϊ��ɫ
    I_bw = I_PM;
    black = abs(I_bw-gray_obj*ones(m,n));
    white = abs(I_bw-gray_back*ones(m,n));
    I_bw = (black >= white)*255;
%     figure;imshow(uint8(I_bw));title('ˮ�׶�ֵ��');
    %% ˮ��С��40��ȥ��
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
    I_bw = ismember(LT, find([S.Area]>45));
%     figure;imshow(I_bw);title('ȥ��Сˮ��')
    %% �߶�С��90����Ϊ��αˮ�ݣ�ȥ��
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
    for i=1:LTnum
        rec_l = ceil(S(i).BoundingBox(1)); % ���ο�����ұ߽���
        rec_r = floor(S(i).BoundingBox(1)+S(i).BoundingBox(3));
        % �ҵ��߽�������Ӧ��ILMcol��RPEcol�����е�����
        ILM_l = find(ILMcol==rec_l);
        ILM_r = find(ILMcol==rec_r);
        RPE_l = find(RPEcol==rec_l);
        RPE_r = find(RPEcol==rec_r);
        
        % ����������������Ӧ��ILM��RPE���С��80������Ϊ��֣���������Ĥ����
        if sum((RPErow(RPE_l:RPE_r)-ILMrow(ILM_l:ILM_r))>90) == 0
            I_bw1 = ismember(LT, find([S.Area]~=S(i).Area));
            I_bw = I_bw & I_bw1;
        end
    end
%     figure;imshow(I_bw);title('ȥ��С��80')
    J(:,:,1) = J(:,:,1)+uint8(I_bw*255);
    J(:,:,2) = J(:,:,2)-uint8(I_bw*255);
    J(:,:,3) = J(:,:,3)-uint8(I_bw*255);
    figure;imshow(J);
% end