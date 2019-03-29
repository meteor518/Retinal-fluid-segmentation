function [ILMrow, ILMcol, ILMnum, RPErow, RPEcol, RPEnum] = ILMRPE_seg(I_bw, I_PM, th)
% I_bw: ��ֵ��ͼ��Ϊ�˵õ�ILM��
% I_PM: �˲�ͼ�����ڵõ����ر仯�ָ�RPE��ͬʱ�жϵõ��ı߽������������Ƿ����
% th: I_PM���ж���ֵ

% ILM...: ILM��ı߽����С��к͵���
% RPE...: RPE��ı߽����С��к͵���

 %% ������ϲ�ILM
% ���ڶ�ֵ��ͼ��ÿ��Ѱ�ҵ�һ����ɫ���ص㣬��ΪILM��
ILMnum = 0;
[m, n] = size(I_bw);
for j=1:n
    for i=1:m
        if I_bw(i,j) == 1   % Ѱ��ÿ�е�һ�������ص㣬��ΪILM
            ILMnum = ILMnum+1;
            ILMrow(ILMnum) = i;  % ����ILM����к��е�����
            ILMcol(ILMnum) = j;            
            break
        end
    end
end
% �����Լ��
ILMrow = Integrity_test(ILMrow,ILMcol,ILMnum);

% ��鲹��������߶Σ����������Ƿ񳬳��߽�
% ���
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

% �ұ�
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

%% �ָ����²�RPE��
% ÿ�лҶ�ֵ�仯������ǰ����һ��ΪILM��RPE�㣬���������ж�������ļ�ΪRPE��
RPEnum = 0;
for j=ILMcol(1):ILMcol(ILMnum)
    RPEnum = RPEnum+1;
    x1 = 1:m;
    y1 = I_PM(x1,j);
  % plot(x1,y1);hold on
    [pksmax,locmax] = findpeaks(y1);   % Ѱ��ÿ�еļ���ֵ;pksmaxΪ������ֵ������ֵ��locmax�Ǽ���ֵ��Ӧ������λ��
    MaxPks = [locmax,pksmax];
    [pksmin,locmin] = findpeaks(-y1);  % Ѱ��ÿ�еļ�Сֵ;pksminΪ����Сֵ������ֵ��locmax�Ǽ�Сֵ��Ӧ������λ��
    MinPks = [locmin,-pksmin];
  % plot(locmax,pksmax,'b+');
  % plot(locmin,-pksmin,'ro');
    Pks = [MaxPks;MinPks];             % �����м�ֵ����ͬһ����Pks
    [pks_value,pks_index] = sort(Pks(:,1));  % �����м�ֵ���������������
    pks_diff = zeros(length(pks_value),1); % pks_diff����������ڼ�ֵ��Ĳ�ֵ����һ�׵����仯�ʣ�

    for i=1:length(pks_value)
        if i==1
            pks_diff(i) = 0;
        else
            pks_diff(i) = Pks(pks_index(i),2)-Pks(pks_index(i-1),2);
        end
    end

    [pksdiff_value,pksdiff_index] = sort(pks_diff); % �Ա仯�ʽ��������ɰ׵��ڼ��仯��������λ�ü�Ϊ�ֽ�
                                                    % ���ɰ׵��ڣ��ڼ���Ϊ��ֵ���ʲ�ֵ��С��ǰ������Ϊ�仯���ģ�һ��ILM��RPE
    if Pks(pks_index(pksdiff_index(1)),1) < Pks(pks_index(pksdiff_index(2)),1)% ȡǰ�������������бȽϣ������ΪRPE�㣬����RPErow�С�RPEcol��
        RPErow(RPEnum) = Pks(pks_index(pksdiff_index(2)),1);
    else
        RPErow(RPEnum) = Pks(pks_index(pksdiff_index(1)),1);
    end

    RPEcol(RPEnum) = j;
end
% �����Լ��
RPErow = Integrity_test(RPErow,RPEcol,RPEnum);
RPErow = RPErow+2;
% ��鲹��������߶Σ����������Ƿ񳬳��߽�
% ���
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

% �ұ�
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