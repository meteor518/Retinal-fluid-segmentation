clear;
clc;

% 对输入图像进行各向异扩散滤波
% 然后利用SFCM分割

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
        I_ori = rgb2gray(I);
    end
    J(:,:,1) = I_ori;
    J(:,:,2) = I_ori;
    J(:,:,3) = I_ori;
    I = double(I_ori);          % I：原图
    
    %% ***************各向异性扩散滤波*****************
    niter = 20;
    lambda = 0.1;
    kappa = 20;
    option = 1;
    I_PM = anisodiff_PM(I,niter, kappa, lambda, option); % I_PM：各向异性扩散滤波后图像
%     figure;imshow(uint8(I_PM));title('PM20_01');

    %% ***************SFCM分割*****************
    cluster_n = 4;
    [center, U, obj_fcn] = sfcm(I_PM, cluster_n, [0.3; 100; 3]);
    maxU = max(U);
    temp = sort(center);
    for i=1:cluster_n
        eval(['cluster',int2str(i),'_index = find(U(',int2str(i),',:) == maxU);']);
        index = find(temp == center(i));
        switch index
            case 1
                color_class = 0;
            case cluster_n 
                color_class = 255;
            otherwise
                color_class = fix(255*(index-1)/(cluster_n-1));
        end
        eval(['I(cluster',int2str(i),'_index) =',int2str(color_class),';']);
    end
    I = uint8(I);
    figure;imshow(I);title('第一次SFCM')
    
    %% ***************ILM和RPE层分割***************
    % 将背景变黑，其他中心变白
    I_bw = I<max(center);
    figure;imshow(I_bw)
    % 去除视网膜中连通域小于200的区域，去噪
    [LT,LTnum] = bwlabel(I_bw);
    S = regionprops(LT,'all');
    I_bw = ismember(LT, find([S.Area]>200));
    figure;imshow(I_bw)
    
    %% 标记ILM和RPE
    [ILMrow, ILMcol, ILMnum, RPErow, RPEcol, RPEnum] = ILMRPE_seg(I_bw, I_PM, max(center));
    
    for i=1:ILMnum
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),1) = 255;  % 把ILM层标记为红色
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),2) = 0;
        J(ILMrow(i)-2:ILMrow(i),ILMcol(i),3) = 0;
    end   
    figure;imshow(J);title('ILM标记')
    
    for i=1:RPEnum
        J(RPErow(i):RPErow(i)+2,RPEcol(i),1) = 255;  % 把RPE层标记为黄色
        J(RPErow(i):RPErow(i)+2,RPEcol(i),2) = 255;
        J(RPErow(i):RPErow(i)+2,RPEcol(i),3) = 0;
    end
%     figure; imshow(J)
 %% ****************SFCM分割水肿*******************
    %% 取出分割的视网膜区域 
    new_PM = I_PM(min(ILMrow):max(RPErow), min(RPEcol):max(RPEcol));
    figure;imshow(uint8(new_PM));
   
    %% 二次SFCM
    cluster_n = 6;
    [center, U, obj_fcn] = sfcm(new_PM, cluster_n, [0.3; 100; 3]);
    maxU = max(U);
    temp = sort(center);
    for i=1:cluster_n
        eval(['cluster',int2str(i),'_index = find(U(',int2str(i),',:) == maxU);']);
        index = find(temp == center(i));
        switch index
            case 1
                color_class = 0;
            case cluster_n 
                color_class = 255;
            otherwise
                color_class = fix(255*(index-1)/(cluster_n-1));
        end
        eval(['new_PM(cluster',int2str(i),'_index) =',int2str(color_class),';']);
    end
    figure;imshow(uint8(new_PM));title('第二次SFCM');
    I_bw = new_PM>=max(center);
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
    figure; imshow(I_bw)
    new_bw = zeros(m,n);
    new_bw(min(ILMrow):max(RPErow), min(RPEcol):max(RPEcol))= I_bw;
    %% 对new_bw图像去背景，ILM-RPE上下左右变为黑色
    %RPE层以下的背景变为黑色
    for j=1:RPEnum
        new_bw(RPErow(j):m,RPEcol(j)) = 0;  
    end
    %ILM层以上的背景变为黑色
    for j=1:ILMnum
        new_bw(1:ILMrow(j),ILMcol(j)) = 0;  
    end
    % 最右点为界，视网膜以右的背景变为黑色
    for j=RPEcol(RPEnum)+1:j
        new_bw(1:m,j) = 0;   
    end
    % 最左点为界，视网膜以右的背景变为黑色
    for j=1:RPEcol(1)-1
        new_bw(1:m,j) = 0;   
    end
    figure;imshow(new_bw);title('水肿标记')
    J(:,:,1) = J(:,:,1)+uint8(new_bw*255);
    J(:,:,2) = J(:,:,2)-uint8(new_bw*255);
    J(:,:,3) = J(:,:,3)-uint8(new_bw*255);
    figure;imshow(J);
% end