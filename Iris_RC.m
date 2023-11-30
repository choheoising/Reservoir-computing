%% Iris Dataset Classification_Signal Generator
clear;
clc;
rng(100);
% -------------------------Parameters---------------------------
num_sample = 150;
num_char = 4;
ML = 4;
N = 16;
% memristor
Vmax = 5;
Vmin = 2;
Vreset = -2;
% pulse
interval = 0.01;
column = 20;
% --------------------------Dataset-----------------------------
% initialize dataset
load('iris_dataset.mat');
rowrank = randperm(size(iris_dataset, 1));
dataset_scuffle = iris_dataset(rowrank,:);
Data = dataset_scuffle(:,1:4);
Label = dataset_scuffle(:,5:7);
save('Label.mat');
% ----------------------Mask Processing-------------------------
% generate mask matrix
Mask = randi([0,1],num_char,ML,N) * 2 - 1;
input_mask = [];
for i = 1:N
    for j = 1:num_sample
        input_mask(i,(j-1)*ML+1:j*ML) = Data(j,:) * Mask(:,:,i);
    end
end
% ----------------------Normalization---------------------------
UL = max(max(input_mask));
DL = min(min(input_mask));
input_norm = (input_mask-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;
% ------------------Input Signal Generator----------------------
% add RESET pulses
temp_1 = reshape(input_norm',ML,[]);
reset = Vreset*ones(1,num_sample*N);
temp_2 = [reset;temp_1];
input_reshape = reshape(temp_2,[],1);
% generate final input signal
input_pulse = [];
zero_pulse = [0];
num_zero = 2;
num_input = 2
for i = 1:size(input_reshape,1)
    for j = 1:num_zero
        input_pulse = [input_pulse;zero_pulse];
    end
    for j = 1:num_input
        input_pulse = [input_pulse;input_reshape(i)];
    end
end
% rearrange
time = colon(0,interval,interval*((size(input_pulse,1))/column-1))';
input_pulse = reshape(input_pulse,size(input_pulse,1)/column,column);
final_input = [];
for k = 1:column
    a = [time,input_pulse(:,k)];
    final_input = [final_input,a];
end
%% Iris Dataset Classification_Train and Test
clear;
clc;
% -------------------------Parameters---------------------------
ML = 4;
N = 16;
% ------------------------Preprocessing-------------------------
% import data
load('memristor_output.mat')
output = flip(memristor_output,2);
% delete RESET current
matrix_1 = reshape(output,[],1);
matrix_2 = reshape(matrix_1,ML+1,[]);
matrix_3 = matrix_2(2:end,:);
matrix_4 = reshape(matrix_3,[],N)';
% initialize output and label
train_state = matrix_4(:,1:ML*120);
test_state = matrix_4(:,ML*120+1:end);
load('Label.mat')
label = Label';
train_label = label(:,1:120);
test_label = label(:,120+1:end);
% --------------------------Train-----------------------------
% states collection
states = [];
for i = 1:120
    a = train_state(:, ML*(i-1)+1:ML*i);
    states(:, i) = a(:);
end
X = [ones(1,120); states];
% linear regression
Wout = train_label*pinv(X);
% ---------------------------Test-----------------------------
% states collection
states = [];
for i = 1:30
    a = test_state(:, ML*(i-1)+1:ML*i);
    states(:,i) = a(:);
end
X = [ones(1,30);states];
% linear regression
out = Wout*X;
% ------------------------Assessment---------------------------
% winner takes all
[Out,index] = max(out,[],1);
Out = index;
[Lb,index] = max(test_label,[],1);
Lb = index;
% accuracy
j=0;
for i = 1:size(Out,2)
    if Out(i) == Lb(i)
        j=j+1;
    end
end
acc = j/size(Out,2);
% ----------------------------Plot-----------------------------
% Figure 1
figure(1);
plot(Lb, '-k*', 'linewidth', 2);
hold on;
plot(Out, '-r*', 'linewidth',1);
axis([0, 30, 1, 3]);
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction');
xlabel('Time Step');
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gca,'ytick',[1 2 3]);
set(gca,'yticklabel',{'setosa' 'versicolor' 'virginica'});
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);
% Figure 2
figure(2);
matrix = [];
for k = 1:3
    a = find(Lb == k);
    b1 = sum(Out(a)==1)/size(a,2);
    b2 = sum(Out(a)==2)/size(a,2);
    b3 = sum(Out(a)==3)/size(a,2);
    c = [b1;b2;b3];
    matrix = [matrix,c];
end
x = [1 2 3];y = [1 2 3];
imagesc(x, y, matrix);
ylabel('Predicted output');
xlabel('Correct output');
set(gca,'ytick',[1 2 3]);
set(gca,'yticklabel',{'setosa' 'versicolor' 'virginica'});
set(gca,'xtick',[1 2 3]);
set(gca,'xticklabel',{'setosa' 'versicolor' 'virginica'});
title(['Acc: ',num2str(acc*100),'%']);
colorbar;
colormap(flipud(hot)); 
set(gca,'FontName', 'Arial', 'FontSize', 15);