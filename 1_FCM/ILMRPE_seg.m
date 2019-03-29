function [ILMrow, ILMcol, ILMnum, RPErow, RPEcol, RPEnum] = ILMRPE_seg(I_bw, I_PM, th)
% I_bw: 二值化图，为了得到ILM层
% I_PM: 滤波图，用于得到像素变化分割RPE，同时判断得到的边界点的左右两侧是否合适
% th: I_PM的判断阈值

% ILM...: ILM层的边界点的行、列和点数
% RPE...: RPE层的边界点的行、列和点数

 %% 标记最上层ILM
% 对于二值化图像，每列寻找第一个白色像素点，记为ILM层
ILMnum = 0;
[m, n] = size(I_bw);
for j=1:n
    for i=1:m
        if I_bw(i,j) == 1   % 寻找每列第一个黑像素点，即为ILM
            ILMnum = ILMnum+1;
            ILMrow(ILMnum) = i;  % 保存ILM层的行和列的坐标
            ILMcol(ILMnum) = j;            
            break
        end
    end
end
% 完整性检测
ILMrow = Integrity_test(ILMrow,ILMcol,ILMnum);

% 检查补完整后的线段，左右两侧是否超出边界
% 左边
temp_num = 0;
for i=1:ILMnum
    if sum(I_PM(ILMrow(i):ILMrow(i)+9,ILMcol(i)))/10 > th
        temp_num = temp_num + 1;
    else
        break
    end
end
ILMrow = ILMrow(temp_num+1:ILMnum);
ILMcol = ILMcol(temp_num+1:ILMnum);
ILMnum = ILMnum - temp_num;

% 右边
temp_num = 0;
for i=ILMnum:-1:1
    if sum(I_PM(ILMrow(i):ILMrow(i)+9,ILMcol(i)))/10 > th
        temp_num = temp_num + 1;
    else
        break
    end
end
ILMnum = ILMnum - temp_num;
ILMrow = ILMrow(1:ILMnum);
ILMcol = ILMcol(1:ILMnum);

%% 分割最下层RPE层
% 每列灰度值变化率最大的前两个一般为ILM和RPE层，根据行数判定行数大的即为RPE层
RPEnum = 0;
for j=ILMcol(1):ILMcol(ILMnum)
    RPEnum = RPEnum+1;
    x1 = 1:m;
    y1 = I_PM(x1,j);
  % plot(x1,y1);hold on
    [pksmax,locmax] = findpeaks(y1);   % 寻找每列的极大值;pksmax为各极大值的像素值，locmax是极大值对应的行数位置
    MaxPks = [locmax,pksmax];
    [pksmin,locmin] = findpeaks(-y1);  % 寻找每列的极小值;pksmin为各极小值的像素值，locmax是极小值对应的行数位置
    MinPks = [locmin,-pksmin];
  % plot(locmax,pksmax,'b+');
  % plot(locmin,-pksmin,'ro');
    Pks = [MaxPks;MinPks];             % 把所有极值存入同一矩阵Pks
    [pks_value,pks_index] = sort(Pks(:,1));  % 按所有极值点的行数进行排序
    pks_diff = zeros(length(pks_value),1); % pks_diff：存放两相邻极值点的差值，即一阶导（变化率）

    for i=1:length(pks_value)
        if i==1
            pks_diff(i) = 0;
        else
            pks_diff(i) = Pks(pks_index(i),2)-Pks(pks_index(i-1),2);
        end
    end

    [pksdiff_value,pksdiff_index] = sort(pks_diff); % 对变化率进行排序，由白到黑即变化降得最快的位置即为分界
                                                    % 因由白到黑，黑减白为负值，故差值最小的前两个即为变化最快的，一般ILM或RPE
    if Pks(pks_index(pksdiff_index(1)),1) < Pks(pks_index(pksdiff_index(2)),1)% 取前两个的行数进行比较，大的判为RPE层，存入RPErow行、RPEcol列
        RPErow(RPEnum) = Pks(pks_index(pksdiff_index(2)),1);
    else
        RPErow(RPEnum) = Pks(pks_index(pksdiff_index(1)),1);
    end

    RPEcol(RPEnum) = j;
end
% 完整性检测
RPErow = Integrity_test(RPErow,RPEcol,RPEnum);
RPErow = RPErow+2;
% 检查补完整后的线段，左右两侧是否超出边界
% 左边
temp_num = 0;
for i=1:RPEnum
    if sum(I_PM(RPErow(i)-2:RPErow(i)+2,RPEcol(i)))/5 > th
        temp_num = temp_num + 1;
    else
        break
    end
end
RPErow = RPErow(temp_num+1:RPEnum);
RPEcol = RPEcol(temp_num+1:RPEnum);
RPEnum = RPEnum - temp_num;

% 右边
temp_num = 0;
for i=RPEnum:-1:1
    if sum(I_PM(RPErow(i)-2:RPErow(i)+2,RPEcol(i)))/5 > th
        temp_num = temp_num + 1;
    else
        break
    end
end
RPEnum = RPEnum - temp_num;
RPErow = RPErow(1:RPEnum);
RPEcol = RPEcol(1:RPEnum);
end