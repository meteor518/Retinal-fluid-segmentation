function output = Integrity_test(edge_rows,edge_cols,num)
%% �߽��߶ε������Լ��
% edge_rowsΪ���б߽��(ILM/RPE)��������
% edge_colsΪ���б߽��(ILM/RPE)��������
% numΪ����Ĥ��������

% �����������Ӱ�죬�ʱ仯���Ĳ�һ�����ǲ㣬����Ҫ���������ڲ���о����жϣ�������ǰһ��������ڱ��������򽫸���ǰһ���ֵ���¸�ֵ�õ�
line_index = 1;                         % ���ڴ���߶ε�����
line_num = 1;                           % ���ڼ���ÿ���������߶εĳ���
line(line_index,1) = edge_rows(1);         % line����һ�д�ÿ���߶���ʼ��
line(line_index,2) = edge_cols(1);         % line���ڶ��д�ÿ���߶���ʼ��

for i=2:num
    if abs(edge_rows(i)-edge_rows(i-1))<3
        line_num = line_num+1;
    else
        line(line_index,3) = edge_rows(i-1);%line�������д�ÿ���߶ν�����
        line(line_index,4) = edge_cols(i-1);%line�������д�ÿ���߶ν�����
        line(line_index,5) = line_num;   %line�������д�ÿ���߶γ���
        line_index = line_index+1;
        line(line_index,1) = edge_rows(i); 
        line(line_index,2) = edge_cols(i);  
        line_num = 1;
    end
    if i==num
        line(line_index,3) = edge_rows(i);
        line(line_index,4) = edge_cols(i);
        line(line_index,5) = line_num; 
    end
end
% ÿ���߶���<40,����ȥ,��Ϊ0�����������Ĵ���line1����
line1_index = 0;
for i=1:line_index
    if line(i,5)<40
        line(i,:) = 0;
    else
        line1_index = line1_index+1;
        line1(line1_index,1) = line(i,1);
        line1(line1_index,2) = line(i,2);
        line1(line1_index,3) = line(i,3);
        line1(line1_index,4) = line(i,4);
    end
end
% ��ȥ�Ĳ�������ǰ�������������߶���ʼ��������������
if line1(line1_index, 4) < num
    edge_rows(line1(line1_index, 4):num) = line1(line1_index, 3);
end
if line1(1, 2) > edge_cols(1)
    edge_rows(edge_cols(1):line1(1, 2)) = line1(1, 1);
end
for i=1:line1_index-1
    for j=line1(i,4):line1(i+1,2)
        edge_rows(j) = round((line1(i,3)-line1(i+1,1))/(line1(i,4)-line1(i+1,2))*(j-line1(i,4))+line1(i,3));
    end
end
output = edge_rows;