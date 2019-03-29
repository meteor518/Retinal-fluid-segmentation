clear;
clc;
%**************************************************************************
%*********************************ˮ����ʾ����*****************************
%**************************************************************************
%% ��ʼ����
% ���ڴ�ÿ����ɫ����ʼ���꣬1�кţ�2-3��ʼ�кţ�4��ɫ��
bluebar0(:,1:4) = 0;  bluebar1(:,1:4) = 0;  bluebar2(:,1:4) = 0;   
bluebar3(:,1:4) = 0;  bluebar4(:,1:4) = 0;  bluebar5(:,1:4) = 0; 
bluebar6(:,1:4) = 0;  bluebar7(:,1:4) = 0;  bluebar8(:,1:4) = 0; 
bluebar9(:,1:4) = 0;  bluebar10(:,1:4) = 0; bluebar11(:,1:4) = 0; 
bluebar12(:,1:4) = 0; bluebar13(:,1:4) = 0; bluebar14(:,1:4) = 0;
bluebar15(:,1:4) = 0; bluebar16(:,1:4) = 0;
bar = zeros(16, 1);     % ���ڼ���ÿ����ɫ������

%% ͼ��Ŀ¼
path_left = '';     % һ��OCT������۵�ͼ
path_right = '';    % һ��OCT���Ҳ�ˮ�׷ָ��Ķ�ֵͼ
I_original = imread('ori.bmp');   % �۵�ԭͼ
I1 = I_original;
%imshow(I_original);hold on
limg_list = dir([path_left, '*.png']); 
rimg_list = dir([path_right, '*.png']);
img_num = length(limg_list);
% num = 8;

for num = 1:img_num
    %% ***************��ͼ***************
    limg_name = limg_list(num).name;
    I_color = imread(strcat(path_left, limg_name));     
    I_color = double(I_color);           % I_color:����ɫͼ
    
    rimg_name = rimg_list(num).name;
    I_bw = imread(strcat(path_right, rimg_name)); % I���Ҳ�Ҷ�ͼ��ˮ�׷ָ��ֵͼ
    
    [m, n] = size(I_bw);                     % m,nΪI�Ĵ�С
    [m1, n1] = size(I_color(:,:,1));      % m1,n1ΪI_color�Ĵ�С
   
    %% **************�õ�ˮ����Ϣ****************
    [bubbleline, LTnum] = get_info(I_bw);
     
    %% **************ӳ��ˮ����Ϣ****************
    % ��ɫͼ�ϱ�ǣ���ɫ�߳����335������
    
    %% �ҵ�ɨ���ߵ�λ��
    colorrow = 0;    % ��¼������ʼ����
    colorcol = 0;    % ��¼������ʼ����
    for j = 1:n1
        for i = 1:m1
            if I_color(i,j,1)==I_color(i,j,2) && I_color(i,j,3)==I_color(i,j,2) % �ҵ������Ŀ�ʼλ��
            else
                colorrow = i;
                colorcol = j;
                break
            end
        end
        if colorrow
            break
        end
    end

    %% ��ˮ�ݸ߶ȷ���������ɫ������ɫ��ӳ�ݵĸ߶ȣ�����17���ȼ�
   [bubbleline_value,bubbleline_index] = sort(bubbleline(:,1));       % ��ˮ�ݵĸ߶�����

    for i=1:bubblenum
        % �ж�ͬһ��ɨ�����ϲ�ͬˮ�׵�ӳ���Ƿ��ص�
        if i>1
            for j=1:i-1
                if bubbleline(bubbleline_index(i),2)>=bubbleline(bubbleline_index(j),3)...
                        ||bubbleline(bubbleline_index(i),3)<=bubbleline(bubbleline_index(j),2)
                else
                    colorrow = colorrow+1;
                    break
                end
            end
        end
        %% 17���ȼ�����
        % 0
        if bubbleline(bubbleline_index(i),1)>=250
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 0;
            I_original(colorrow,col1:col2,3) = 255;
            bar0 = bar0+1;
            bluebar0(bar0,1) = colorrow;
            bluebar0(bar0,2) = col1;
            bluebar0(bar0,3) = col2;
            bluebar0(bar0,4) = 1;
        end
        % 1
        if bubbleline(bubbleline_index(i),1)>=240 && bubbleline(bubbleline_index(i),1)<250
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 16;
            I_original(colorrow,col1:col2,3) = 255;
            bar1 = bar1+1;
            bluebar1(bar1,1) = colorrow;
            bluebar1(bar1,2) = col1;
            bluebar1(bar1,3) = col2;
            bluebar1(bar1,4) = 2;
        end
        % 2
        if bubbleline(bubbleline_index(i),1)>=230 && bubbleline(bubbleline_index(i),1)<240
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 32;
            I_original(colorrow,col1:col2,3) = 255;
            bar2 = bar2+1;
            bluebar2(bar2,1) = colorrow;
            bluebar2(bar2,2) = col1;
            bluebar2(bar2,3) = col2;
            bluebar2(bar2,4) = 3;
        end
        % 3
        if bubbleline(bubbleline_index(i),1)>=220 && bubbleline(bubbleline_index(i),1)<230
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 48;
            I_original(colorrow,col1:col2,3) = 255;
            bar3 = bar3+1;
            bluebar3(bar3,1) = colorrow;
            bluebar3(bar3,2) = col1;
            bluebar3(bar3,3) = col2;
            bluebar3(bar3,4) = 4;
        end
        % 4
        if bubbleline(bubbleline_index(i),1)>=210 && bubbleline(bubbleline_index(i),1)<220
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 64;
            I_original(colorrow,col1:col2,3) = 255;
            bar4 = bar4+1;
            bluebar4(bar4,1) = colorrow;
            bluebar4(bar4,2) = col1;
            bluebar4(bar4,3) = col2;
            bluebar4(bar4,4) = 5;
        end
        % 5
        if bubbleline(bubbleline_index(i),1)>=200 && bubbleline(bubbleline_index(i),1)<210
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 80;
            I_original(colorrow,col1:col2,3) = 255;
            bar5 = bar5+1;
            bluebar5(bar5,1) = colorrow;
            bluebar5(bar5,2) = col1;
            bluebar5(bar5,3) = col2;
            bluebar5(bar5,4) = 6;
        end
        % 6
        if bubbleline(bubbleline_index(i),1)>=190 && bubbleline(bubbleline_index(i),1)<200
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 96;
            I_original(colorrow,col1:col2,3) = 255;
            bar6 = bar6+1;
            bluebar6(bar6,1) = colorrow;
            bluebar6(bar6,2) = col1;
            bluebar6(bar6,3) = col2;
            bluebar6(bar6,4) = 7;
        end
        % 7
        if bubbleline(bubbleline_index(i),1)>=180 && bubbleline(bubbleline_index(i),1)<190
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 112;
            I_original(colorrow,col1:col2,3) = 255;
            bar7 = bar7+1;
            bluebar7(bar7,1) = colorrow;
            bluebar7(bar7,2) = col1;
            bluebar7(bar7,3) = col2;
            bluebar7(bar7,4) = 8;
        end
        % 8
        if bubbleline(bubbleline_index(i),1)>=170 && bubbleline(bubbleline_index(i),1)<180
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 128;
            I_original(colorrow,col1:col2,3) = 255;
            bar8 = bar8+1;
            bluebar8(bar8,1) = colorrow;
            bluebar8(bar8,2) = col1;
            bluebar8(bar8,3) = col2;
            bluebar8(bar8,4) = 9;
        end
        % 9
        if bubbleline(bubbleline_index(i),1)>=160 && bubbleline(bubbleline_index(i),1)<170
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 144;
            I_original(colorrow,col1:col2,3) = 255;
            bar9 = bar9+1;
            bluebar9(bar9,1) = colorrow;
            bluebar9(bar9,2) = col1;
            bluebar9(bar9,3) = col2;
            bluebar9(bar9,4) = 10;
        end
        % 10
        if bubbleline(bubbleline_index(i),1)>=150 && bubbleline(bubbleline_index(i),1)<160
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 160;
            I_original(colorrow,col1:col2,3) = 255;
            bar10 = bar10+1;
            bluebar10(bar10,1) = colorrow;
            bluebar10(bar10,2) = col1;
            bluebar10(bar10,3) = col2;
            bluebar10(bar10,4) = 11;
        end
        % 11
        if bubbleline(bubbleline_index(i),1)>=140 && bubbleline(bubbleline_index(i),1)<150
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 176;
            I_original(colorrow,col1:col2,3) = 255;
            bar11 = bar11+1;
            bluebar11(bar11,1) = colorrow;
            bluebar11(bar11,2) = col1;
            bluebar11(bar11,3) = col2;
            bluebar11(bar11,4) = 12;
        end
        % 12
        if bubbleline(bubbleline_index(i),1)>=130 && bubbleline(bubbleline_index(i),1)<140
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 192;
            I_original(colorrow,col1:col2,3) = 255;
            bar12 = bar12+1;
            bluebar12(bar12,1) = colorrow;
            bluebar12(bar12,2) = col1;
            bluebar12(bar12,3) = col2;
            bluebar12(bar12,4) = 13;
        end
        % 13
        if bubbleline(bubbleline_index(i),1)>=120 && bubbleline(bubbleline_index(i),1)<130
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 208;
            I_original(colorrow,col1:col2,3) = 255;
            bar13 = bar13+1;
            bluebar13(bar13,1) = colorrow;
            bluebar13(bar13,2) = col1;
            bluebar13(bar13,3) = col2;
            bluebar13(bar13,4) = 14;
        end
        % 14
        if bubbleline(bubbleline_index(i),1)>=110 && bubbleline(bubbleline_index(i),1)<120
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 224;
            I_original(colorrow,col1:col2,3) = 255;
            bar14 = bar14+1;
            bluebar14(bar14,1) = colorrow;
            bluebar14(bar14,2) = col1;
            bluebar14(bar14,3) = col2;
            bluebar14(bar14,4) = 15;
        end
        % 15
        if bubbleline(bubbleline_index(i),1)>=100 && bubbleline(bubbleline_index(i),1)<110
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 240;
            I_original(colorrow,col1:col2,3) = 255;
            bar15 = bar15+1;
            bluebar15(bar15,1) = colorrow;
            bluebar15(bar15,2) = col1;
            bluebar15(bar15,3) = col2;
            bluebar15(bar15,4) = 16;
        end
        % 16
        if bubbleline(bubbleline_index(i),1)<100
            col1 = round(colorcol+bubbleline(bubbleline_index(i),2)/n*335);
            col2 = round(colorcol+bubbleline(bubbleline_index(i),3)/n*335);
            I_original(colorrow,col1:col2,1) = 0;
            I_original(colorrow,col1:col2,2) = 256;
            I_original(colorrow,col1:col2,3) = 255;
            bar16 = bar16+1;
            bluebar16(bar16,1) = colorrow;
            bluebar16(bar16,2) = col1;
            bluebar16(bar16,3) = col2;
            bluebar16(bar16,4) = 17;
        end
    end
    clearvars -except path I_original I1 bar0 bar1 bar2 bar3 bar4 bar5 bar6 bar7 bar8 bar9 bar10 bar11 bar12 bar13 bar14 bar15 bar16 ...
        bluebar0 bluebar1 bluebar2 bluebar3 bluebar4 bluebar5 bluebar6 bluebar7 bluebar8 bluebar9 bluebar10 bluebar11 bluebar12 bluebar13 ...
        bluebar14 bluebar15 bluebar16 
end
figure; imshow(I_original); hold on

%% **************ˮ����ʾ****************

% �Ѹ�����ɫ������ɫ�ž���ϳ�һ����ɫ������
bluebar = [bluebar0;bluebar1;bluebar2;bluebar3;bluebar4;bluebar5;...
    bluebar6;bluebar7;bluebar8;bluebar9;bluebar10;bluebar11;bluebar12;
    bluebar13;bluebar14;bluebar15;bluebar16];
[bm,bn] = size(bluebar);                    % bm,bn:������ɫ������Ĵ�С
[bar_value,bar_index] = sort(bluebar(:,1)); % ����ɫ����������������
barcol = 1;         % barcol:���ڽ���ɫ����תΪ��ͬ�����еľ���ʱ�������У�ͬһ�߶ȵ���ɫ������ͬ4��
barrow = 1;         % barrow:���ڽ���ɫ����תΪ��ͬ�����еľ���ʱ�������У�
for i=1:bm
    if bar_value(i)>0
        blueline(barrow,barcol:barcol+3) = bluebar(bar_index(i),1:4);   % ȡ������ĵ�һ���ߣ�blueline�����ڴ����ɨ���߸߶�����ľ���
        barstart = i;
        break
    end
end

for i=barstart+1:bm
    if bar_value(i)-bar_value(i-1) > 1    % �����ǰ��ɫ������һ�������1����ʾ������ͬһ�߶ȣ�������ָ��+4����ָ��ص�����
        barcol = barcol+4;
        barrow = 1;
        blueline(barrow,barcol:barcol+3) = bluebar(bar_index(i),1:4);
    else
        barrow = barrow+1;                 % ����ͬһ�߶ȵģ���ָ�벻�䣬��ָ��+1
        blueline(barrow,barcol:barcol+3) = bluebar(bar_index(i),1:4);
    end
end

%% �����б�ʾˮ�ݵ���ɫ�ߣ���������ɨ���߼�����߸��ݸ߶ȡ��Ƿ��ص�����Ȳ���б������������ӣ��γ�ˮ������

[blm,bln] = size(blueline);     % blueline����ɨ���д洢����ɫ�����꣬blm,blnΪblueline�ľ����С
temp(1,:) = 0;                  % temp,temp1Ϊ����ָ�룬�����ж���������ɨ��������ɫ���Ƿ�����ͬһ��ˮ��
temp1(1,:) = 0;
color(1,1,1) = 0;               % color���ڸ�ֵˮ����ʾ��ɫ
color(1,1,2) = 200;
color(1,1,3) = 255;

edge_point(:,1:3) = 0;          % edge_point���ڴ�����ͬһ���ݵ���������

Imagebubble1 = I1;               % Imagebubble���洢�����γ�ˮ�ݺ��ͼ��
Imagebubble_num = 1;            % Imagebubble_num������ˮ��ͼ�ĸ���
I_original = I1;

for j=4:4:bln           % bubbleline��ÿ���д����һ��ɨ���ߵ���Ϣ�����Բ���Ϊ4
%    j=4;
    flag = 1;           % flag�������˳�ѭ�������ҵ���һɨ��������ͬһ��ˮ�ݵ���ɫ����flag=0�����������бȽ�
    [value,index] = sort(blueline(:,j));
%      i=3;
    for i=1:blm
        flag = 1;
        point = 1;
        edge_point(:,1:3) = 0;
        I_previous = I_original;
        I_original = I1;
        if value(i)>0               % ȡÿһ��ɨ���������ߣ�����ɫ�Ŵ�С����ȡ
            temp = blueline(index(i),j-3:j);  % temp��ǰ����
            color(1,1,1) = color(1,1,1)+10;
            blueline(index(i),j-3:j) = 0;     % ȡ��������������ɫ����ԭ�����������
            edge_point(point,1) = temp(1)-round((temp(3)-temp(2))*3/8);     % ÿ�ζ�һ���������Զ���һ���㣬�γɷ����
            edge_point(point,2:3) = round((temp(2)+temp(3))/2);
            point = point+1;
            edge_point(point,:) = temp(1:3);
            k = j+4;
            while k<=bln
                [value1,index1] = sort(blueline(:,k));
                for i1=1:blm
                    if value1(i1)>0
                        temp1 = blueline(index1(i1),k-3:k);   % ȡ����һ����С��ɫ��ŵ���
                        break
                    end
                end
                if temp1
                    for i3=i1:blm
                        temp1 = blueline(index1(i3),k-3:k);
                        if temp1(4)==temp(4)
                            if temp(2)>=temp1(3)||temp(3)<=temp1(2)
                                if i3==blm
                                    point = point+1;
                                    edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                    edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                    flag = 0;
                                    break
                                end
                            else if abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<60||abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<min(temp(3)-temp(2),temp1(3)-temp1(2))                    
                                    point = point+1;
                                    edge_point(point,:) = temp1(1:3);
                                    if k==bln
                                        point = point+1;
                                        edge_point(point,1) = temp1(1)+round((temp1(3)-temp1(2))*3/8);
                                        edge_point(point,2:3) = round((temp1(2)+temp1(3))/2);
                                    end
                                    blueline(index1(i3),k-3:k) = 0;
                                    temp = temp1;
                                    break
                                else
                                    if i3==blm
                                        point = point+1;
                                        edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                        edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                        flag = 0;
                                        break
                                    end
                                end
                            end
                        else if abs(temp1(4)-temp(4))==1
                                if temp(2)>=temp1(3)||temp(3)<=temp1(2)
                                    if i3==blm
                                        point = point+1;
                                        edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                        edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                        flag = 0;
                                        break
                                    end
                                else if abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<60||abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<min(temp(3)-temp(2),temp1(3)-temp1(2))
                                        point = point+1;
                                        edge_point(point,:) = temp1(1:3);
                                        if k==bln
                                            point = point+1;
                                            edge_point(point,1) = temp1(1)+round((temp1(3)-temp1(2))*3/8);
                                            edge_point(point,2:3) = round((temp1(2)+temp1(3))/2);
                                        end
                                        blueline(index1(i3),k-3:k) = 0;
                                        temp = temp1;
                                        break
                                    else
                                        if i3==blm
                                            point = point+1;
                                            edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                            edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                            flag = 0;
                                            break
                                        end
                                    end
                                end
                            else if abs(temp1(4)-temp(4))==2
                                    if temp(2)>=temp1(3)||temp(3)<=temp1(2)
                                        if i3==blm
                                            point = point+1;
                                            edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                            edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                            flag = 0;
                                            break
                                        end
                                    else if abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<60||abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<min(temp(3)-temp(2),temp1(3)-temp1(2))
                                            point = point+1;
                                            edge_point(point,:) = temp1(1:3);
                                            if k==bln
                                                point = point+1;
                                                edge_point(point,1) = temp1(1)+round((temp1(3)-temp1(2))*3/8);
                                                edge_point(point,2:3) = round((temp1(2)+temp1(3))/2);
                                            end
                                            blueline(index1(i3),k-3:k) = 0;
                                            temp = temp1;
                                            break
                                        else
                                            if i3==blm
                                                point = point+1;
                                                edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                                edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                                flag = 0;
                                                break
                                            end
                                        end
                                    end
                                else if abs(temp1(4)-temp(4))==3
                                        if temp(2)>=temp1(3)||temp(3)<=temp1(2)
                                            if i3==blm
                                                point = point+1;
                                                edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                                edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                                flag = 0;
                                                break
                                            end
                                        else if abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<60||abs(temp(3)-temp(2)-(temp1(3)-temp1(2)))<min(temp(3)-temp(2),temp1(3)-temp1(2))
                                                point = point+1;
                                                edge_point(point,:) = temp1(1:3);
                                                if k==bln
                                                    point = point+1;
                                                    edge_point(point,1) = temp1(1)+round((temp1(3)-temp1(2))*3/8);
                                                    edge_point(point,2:3) = round((temp1(2)+temp1(3))/2);
                                                end
                                                blueline(index1(i3),k-3:k) = 0;
                                                temp = temp1;
                                                break
                                            else
                                                if i3==blm
                                                    point = point+1;
                                                    edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                                    edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                                    flag = 0;
                                                    break
                                                end
                                            end
                                        end
                                    else
                                        point = point+1;
                                        edge_point(point,1) = temp(1)+round((temp(3)-temp(2))*3/8);
                                        edge_point(point,2:3) = round((temp(2)+temp(3))/2);
                                        flag = 0;
                                        break
                                    end
                                end
                            end
                        end
                    end
                end
                if flag
                    k = k+4;
                else
                    break
                end
            end
        end
        % ��ֵ����Ե��⻬����
        if edge_point(1,1)
            x1 = edge_point(1,1);
            for ii=2:length(edge_point(:,1))
                if edge_point(ii,1)~=0
                    x1 = [x1;edge_point(ii,1)];
                end
            end
            y1 = edge_point(1:length(x1),2);
            y2 = edge_point(1:length(x1),3);
            plot(y1,x1,'or');
            hold on;
            plot(y2,x1,'or');
            hold on;
            Row = min(x1):1:max(x1);
            Col1 = interp1(x1,y1,Row,'PCHIP');
            Col2 = interp1(x1,y2,Row,'PCHIP');
            plot(Col1,Row,'g');
            hold on
            plot(Col2,Row,'g');
            % ������������ֵ��ɫ
            for color_i=2:length(Row)-1
                for color_j=round(Col1(color_i)):round(Col2(color_i))
                I_original(Row(color_i),color_j,:) = color;
                end
            end
        end
%         figure(2);imshow(I_original);
        if I_previous==I_original
            previous_flag = 1;      % previous_flag:�����жϵ�ǰͼ���Ƿ���ǰһ��һ������һ���������µ�ˮ�ݻ���û��ˮ���γ�
        else
            previous_flag = 0;      
        end
        if I_original==I1
            I1_flag = 1;            % I1_flag:�����жϵ�ǰͼ���Ƿ���ԭͼһ����һ��������ˮ��
        else
            I1_flag = 0;
        end
        if previous_flag||I1_flag
        else
            Imagebubble_num = Imagebubble_num+1;
            assignin('base',['Imagebubble',num2str(Imagebubble_num)],I_original);
%             Imagebubble = [Imagebubble,I_original];  % ��һ��Ϊԭͼ���������δ����µ�ˮ���γɵ�ͼ
%             Imagebubble_num = Imagebubble_num+1;
        end
    end
end
% for k = 1:Imagebubble_num
%             Imagebubble = eval(strcat('Imagebubble',num2str(k)));
%             figure(k);imshow(Imagebubble);
% end
%% ��ÿ��ͼ��ˮ�������е����ںϣ�͸����ʾ

[m,n] = size(I1(:,:,1));
Image_final = uint8(zeros(m,n,3));
w = zeros(m,n);             % ��ͼ��Ȩ��
sum = zeros(m,n);           % ÿ�����ˮ��ͼ��ź�
for i=1:m
    for j=1:n     
        TrueFalse = ones(1,Imagebubble_num);  % �����ж�����ͼ�ĸõ��Ƿ�Ϊˮ����������Ϊ1���ǵ�Ϊ0
        zero_num = 0;                           % ���ڼ���Ϊ0����Ŀ
        for k=2:Imagebubble_num
            Imagebubble = eval(strcat('Imagebubble',num2str(k)));
            if Imagebubble(i,j,1)==Imagebubble(i,j,2) && Imagebubble(i,j,2)==Imagebubble(i,j,3)
                TrueFalse(1,k) = 1;
            else
                zero_num = zero_num+1;
                TrueFalse(1,k) = 0;
                sum(i,j) = sum(i,j)+k;
            end
        end
        if zero_num<=5
            w(i,j) = 0.7/4*(5-zero_num);
        else
            w(i,j) = 0;
        end
    end
end
assignin('base',['W',num2str(0)],w);
w0 = eval(strcat('W',num2str(0)));

for k=2:Imagebubble_num
    Imagebubble = eval(strcat('Imagebubble',num2str(k)));
    for i=1:m
        for j=1:n
            if Imagebubble(i,j,1)==Imagebubble(i,j,2) && Imagebubble(i,j,2)==Imagebubble(i,j,3)
                w(i,j) = 0;
            else
                w(i,j) = (1-w0(i,j))*k/sum(i,j);
            end
        end
    end
    assignin('base',['W',num2str(k-1)],w);
end

for k =1:Imagebubble_num
    Imagebubble = eval(strcat('Imagebubble',num2str(k)));
    W = eval(strcat('W',num2str(k-1)));

    Image_final1(:,:,1) = Imagebubble(:,:,1) .* W;
    Image_final1(:,:,2) = Imagebubble(:,:,2) .* W;
    Image_final1(:,:,3) = Imagebubble(:,:,3) .* W;
    
    figure(k);imshow(uint8(Image_final1));
    Image_final = Image_final +Image_final1;
end
figure(k+1);imshow(uint8(Image_final));