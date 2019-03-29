clear;
clc;

% 对输入图像进行各向异性扩散滤波
% 然后利用灰度和梯度变化进行ILM和RPE层分割
% 最后阈值分割

path = 'F:\研究僧科研\7毕业论文实验\image\test\';
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
    I = double(I);          % I：原图
    
    %% ***************各向异性扩散滤波*****************
    niter = 60;
    lambda = 0.1;
    kappa = 60;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM：各向异性扩散滤波后图像
%     figure;imshow(uint8(I_PM));title('PM20_01');

    %% ****************分割ILM和RPE*******************
    gray_deback = 180;      % gray_deback:更接近于目标的灰度值，用于去除背景噪声
    gray_back = 255;        % gray_back:背景灰度域值
    gray_obj = 190;         % gray_obj：目标更精确二值化

    %% 第一次二值化
    % 接近于灰色值的即为目标区，标为黑色，背景色为白色
    % 二值化，为了得到ILM和RPE层后去除背景噪声
    I_bw = I_PM;
    black = abs(I_bw-gray_deback*ones(m,n));
    white = abs(I_bw-gray_back*ones(m,n));
    I_bw = (black < white) * 255;
    % 去除面积小于200的噪声点
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
%     figure;imshow(LT);title('二值化');hold on
%     for i=1:LTnum
%         rectangle('position',S(i).BoundingBox,'edgecolor','r');
%     end 
    I_bw = ismember(LT, find([S.Area]>200));
%     figure;imshow(I_bw);title('二值化');
   
    %% 标记最上层ILM和RPE
    [ILMrow, ILMcol, ILMnum, RPErow, RPEcol, RPEnum] = ILMRPE_seg(I_bw, I_PM, gray_deback);
    
    for i=1:ILMnum
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),1) = 255;  % 把ILM层标记为红色
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),2) = 0;
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),3) = 0;
    end   
%     figure;imshow(uint8(J));title('ILM');
    
    for i=1:RPEnum
        J(RPErow(i):RPErow(i)+2,RPEcol(i),1) = 255;  % 把RPE层标记为黄色
        J(RPErow(i):RPErow(i)+2,RPEcol(i),2) = 255;
        J(RPErow(i):RPErow(i)+2,RPEcol(i),3) = 0;
    end
%     figure; imshow(J)
    
    %% ****************阈值分割水肿*******************
    % 对I_PM图像去背景噪声，上下左右变为黑色
    %RPE层以下的背景变为黑色
    for j=1:RPEnum
        I_PM(RPErow(j):m,RPEcol(j)) = 0;  
    end
    %ILM层以上的背景变为黑色
    for j=1:RPEnum
        I_PM(1:ILMrow(j),RPEcol(j)) = 0;  
    end
    % 最右点为界，视网膜以右的背景变为黑色
    for j=RPEcol(RPEnum)+1:j
        I_PM(1:m,j) = 0;   
    end
    % 最左点为界，视网膜以右的背景变为黑色
    for j=1:RPEcol(1)-1
        I_PM(1:m,j) = 0;   
    end
%     figure;imshow(uint8(I_PM));title('去噪声后的I');
   
    % 二次二值化，准确分割目标区，水泡为白色
    I_bw = I_PM;
    black = abs(I_bw-gray_obj*ones(m,n));
    white = abs(I_bw-gray_back*ones(m,n));
    I_bw = (black >= white)*255;
%     figure;imshow(uint8(I_bw));title('水肿二值化');
    %% 水泡小于40的去除
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
    I_bw = ismember(LT, find([S.Area]>45));
%     figure;imshow(I_bw);title('去除小水泡')
    %% 高度小于90的认为是伪水泡，去除
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
    for i=1:LTnum
        rec_l = ceil(S(i).BoundingBox(1)); % 矩形框的左右边界列
        rec_r = floor(S(i).BoundingBox(1)+S(i).BoundingBox(3));
        % 找到边界列所对应的ILMcol和RPEcol数组中的索引
        ILM_l = find(ILMcol==rec_l);
        ILM_r = find(ILMcol==rec_r);
        RPE_l = find(RPEcol==rec_l);
        RPE_r = find(RPEcol==rec_r);
        
        % 如果整个矩阵框所对应的ILM和RPE层厚都小于80，则视为误分，正常视网膜区域
        if sum((RPErow(RPE_l:RPE_r)-ILMrow(ILM_l:ILM_r))>90) == 0
            I_bw1 = ismember(LT, find([S.Area]~=S(i).Area));
            I_bw = I_bw & I_bw1;
        end
    end
%     figure;imshow(I_bw);title('去除小于80')
    J(:,:,1) = J(:,:,1)+uint8(I_bw*255);
    J(:,:,2) = J(:,:,2)-uint8(I_bw*255);
    J(:,:,3) = J(:,:,3)-uint8(I_bw*255);
    figure;imshow(J);
% end