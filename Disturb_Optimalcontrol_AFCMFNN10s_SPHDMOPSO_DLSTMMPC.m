%%%% AFNN-MOPSO-UtopiaDecision-SOLSTMMPC
%%%% 作者：孙剑
%%%% 时间：2023.12.25

%% 清空环境
tic;clc;clear;close all;format compact

load workplace_KPI.mat
%%MPC gradient method

%    控制初始化
load workplace_DLSTM.mat

test_xx1=[];
test_yo1=[];
test_xx2=[];
test_yo2=[];

test_xx11=[];
test_yo11=[];
test_xx22=[];
test_yo22=[];

ActualOxygen=[];
ActualNOx=[];
ActualEC=[];
BestOxygen=[];
BestNOx=[];
BestEC=[];

ts=10;

  u1_1=13.2702; 
   u1_2=u1_1; 
 u2_1=28.5153;
  u2_2=u2_1;
 u3_1=20.8449;
  u3_2=u3_1;
  y_1 = 5;
  y_2 = 5;
  

%     u1_1=13.8217; 
%    u1_2=u1_1; 
%  u2_1=28.5580;
%   u2_2=u2_1;
%  u3_1=20.8170;
%   u3_2=u3_1;
%   y_1 = 5.0319;
%   y_2 = 5.0319;
  
   U_Y=[y_1 y_2];
   U_X=[u1_1 u2_1 u3_1 u1_2 u2_2 u3_2];
   X = [U_Y U_X];

Hp =5;%预测时域
Hc = 1;%控制时域
y_err_last=0;


yite=0.02;%梯度优化学习率
% Ro1=1;%权重系数
% Ro2=100;
Ro11=5;%权重系数
Ro21=1;
Ro12=1;%权重系数
Ro22=1;
Ro13=1;%权重系数
Ro23=1;

% yite=0.1;%梯度优化学习率
% Ro1_1=8;%权重系数
% Ro2_1=1;
% Ro1_2=1;%权重系数
% Ro2_2=1;
% Ro1_3=1;%权重系数
% Ro2_3=1;
% y_rel=0;

y_rel= 5;

% test_xx=Xtrain(1100:end,:);
% test_yo=output_train(1100:end,:);
% test_xx=Xtrain;
% test_yo=output_train;
test_xx=[];
test_yo=[];
flag_n=0;
tt=0;
jacobian_matrix1=zeros(Hp,Hp);
jacobian_matrix2=zeros(Hp,Hp);
jacobian_matrix3=zeros(Hp,Hp);
features(:,3)=mean5_3(features(:,3),20);
Flow_second_air=features(:,3);

    u1_1_all=[];
    u1_2_all=[];
    u1_3_all=[];
    
    gama =0.005;
    Error=[];
    E0=0.05;
    
    %优化周期10s
    %控制周期2s
    times_interval=5;
    time1=[];
    
            tt_warm_start=[];
    Xtrain_y_err_last=[];
    Ytrain_y_err_last=[];
    
%%%%%%
%数据加载，在线更新
for t=1:size(XTest_NOx,2)
    t
    time1(t) = t*ts;

    x1 = XTest_NOx(:,t);          %初值
ym_NOx(t)=FNN_out(x1,Center_NOx,Width_NOx,W_NOx);
    
    x2 = XTest_CE(:,t);          %初值

    ym_CE(t) = FNN_out(x2,Center_CE,Width_CE,W_CE);
    

         
    %-------------------------优化-----------------------------------
    if mod(t,18)==1
        %%测试优化烟气含氧量%%%%%
%-----------------------------------MOPSO----------------------------------
[bestOxygen,bestNOx,bestCE] = AFNN_SPHDmopsoOxygenForMPC(input_test_NOx(t,:),input_test_CE(t,:),Center_NOx,Width_NOx,W_NOx,inputps_NOx,outputps_NOx,Center_CE,Width_CE,W_CE,inputps_CE,outputps_CE);
%-----------------------------------MOPSO----------------------------------


%网络输出反归一化
         bestNOx=(outputps_NOx.max-outputps_NOx.min).*bestNOx+outputps_NOx.min;
bestCE=(outputps_CE.max-outputps_CE.min).*bestCE+outputps_CE.min;
        
        
        ActualOxygen=[ActualOxygen input_test_CE(t,end)];
        ActualNOx=[ActualNOx output_test_NOx(t)];
        ActualEC=[ActualEC output_test_CE(t)];
        BestOxygen=[BestOxygen bestOxygen(1)];
        BestNOx=[BestNOx bestNOx(1)];
        BestEC=[BestEC bestCE(1)];
        
    else
    ActualOxygen=[ActualOxygen ActualOxygen(end)];
        ActualNOx=[ActualNOx ActualNOx(end)];
        ActualEC=[ActualEC ActualEC(end)];
        BestOxygen=[BestOxygen BestOxygen(end)];
        BestNOx=[BestNOx BestNOx(end)];
        BestEC=[BestEC BestEC(end)];
    end
    
%-----------------控制器作用-------------------------------------
for tt=((t-1)*times_interval+1):(t*times_interval)
        Ref_Hp = bestOxygen*ones(1,Hp);
            %%陈进东
            Ref_Hp = 0.9*Ref_Hp +0.1*y_rel(end);
%     fitness(i)=sum(abs(YR-y).^2)+10*(pop(i,1)-A).^2;%?为控制系统的鲁棒性和收敛性相关的调整因子,
% %%% 将二次风流量作为干扰
u3_1 = Flow_second_air(t);
u(:,tt)=[u1_1 u2_1 u3_1];
% 系统输出
%反馈校正
%% 训练样本预测
TrainMemFunUnitOut=[]; 
TrainSamIn = (u(:,tt)-inputps.min)./(inputps.max-inputps.min);
    SamIn=TrainSamIn;
    % 隶属函数层，模糊化
    for i=1:InDim
        for j=1:RuleNum
            TrainMemFunUnitOut(i,j)=exp(-((SamIn(i)-Center(i,j))^2)/(Width(i,j)^2));
        end
    end
    % 规则层
    TrainRuleUnitOut=prod(TrainMemFunUnitOut); %规则层输出
    % 输出层
    TrainRuleUnitOutSum=sum(TrainRuleUnitOut); %规则层输出求和
    TrainRuleValue=TrainRuleUnitOut./TrainRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
    TrainNetOut=TrainRuleValue*W; %输出层输出，即网络输出

y_rel(tt)=(outputps.max-outputps.min).*TrainNetOut+outputps.min;

X=[y_1 y_2 u1_1 u2_1 u3_1 u1_2 u2_2 u3_2];

  for k=1:Hp
%       X=[y_1 y_2 u1_1 u2_1 u3_1 u1_2 u2_2 u3_2];
        % 归一化
        x=mapminmax('apply',X',inputps_NARX);%数据归一化
    test_final=x;
    train_data = test_final;
%前馈
m=data_num+k;
gate=tanh(test_final'*weight_input_x.*MASK+pre_h_state(:,m-1)'*(weight_input_h.*MASK.*MASK')+bias_gate.*MASK);
input_gate_input=test_final'*weight_inputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_inputgate_c.*MASK.*MASK')+bias_input_gate.*MASK;
forget_gate_input=test_final'*weight_forgetgate_x.*MASK+pre_h_state(:,m-1)'*(weight_forgetgate_c.*MASK.*MASK')+bias_forget_gate.*MASK;
output_gate_input=test_final'*weight_outputgate_x.*MASK+pre_h_state(:,m-1)'*(weight_outputgate_c.*MASK.*MASK')+bias_output_gate.*MASK;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
        pre_h_state(:,m)=tanh(cell_state(:,m)').*output_gate;
        ym(m)= pre_h_state(:,m)'*(weight_preh_h.*MASK');
 
        %反归一化
        y_pred(k)=mapminmax('reverse',ym(m),outputps_NARX);%网络预测数据
        
        YYY(k)=y_pred(k)+y_err_last(end);
        
        %%jacobian信息
% du{k} = ((ones(size(cell_state(:,m)))-tanh(cell_state(:,m)).^2)'.*(exp(-input_gate_input).*(input_gate.^2).*weight_inputgate_x.*gate+input_gate.*(ones(size(gate))-(gate).^2).*weight_input_x+cell_state(:,m-1)'.*exp(-forget_gate_input).*(forget_gate.^2).*weight_forgetgate_x).*output_gate+tanh(cell_state(:,m)').*(exp(-output_gate_input).*(output_gate.^2).*weight_outputgate_x))*weight_preh_h;
du{k} = ((ones(size(cell_state(:,m)))-tanh(cell_state(:,m)).^2)'.*(exp(-input_gate_input).*(input_gate.^2).*weight_inputgate_x.*MASK.*gate+input_gate.*(ones(size(gate))-(gate).^2).*weight_input_x.*MASK+cell_state(:,m-1)'.*exp(-forget_gate_input).*(forget_gate.^2).*weight_forgetgate_x.*MASK).*output_gate+tanh(cell_state(:,m)').*(exp(-output_gate_input).*(output_gate.^2).*weight_outputgate_x.*MASK))*(weight_preh_h.*MASK');

dyy(k)=du{k}(1);
dyu1(k)=du{k}(3);
dyu2(k)=du{k}(4);
dyu3(k)=du{k}(5);
        %         u_2=u_1;
        %         u_1 = u_1;
        y_2=y_1;
        y_1=YY(k);
    end
    
   
    for i=1:Hp
        for j=1:Hp
            if j==i
                jacobian_matrix1(i,j)=dyu1(i);
            elseif j>i
                jacobian_matrix1(i,j)=jacobian_matrix1(i,j-1)*(1+dyy(j-1));
            end
        end
    end
    
        for i=1:Hp
        for j=1:Hp
            if j==i
                jacobian_matrix2(i,j)=dyu2(i);
            elseif j>i
                jacobian_matrix2(i,j)=jacobian_matrix2(i,j-1)*(1+dyy(j-1));
            end
        end
        end
    
            for i=1:Hp
        for j=1:Hp
            if j==i
                jacobian_matrix3(i,j)=dyu3(i);
            elseif j>i
                jacobian_matrix3(i,j)=jacobian_matrix3(i,j-1)*(1+dyy(j-1));
            end
        end
    end
    
    y_rel(end)
    
  
%     if t==15
%         aa=1;
%     end
%     
%     if t>59
%         aa=1;
%     end
%     
%     if t>200
%         aa=1;
%     end
%     
%     if t>194
%         aa=1;
%     end

if tt>4
    y_err_last(tt) = y_rel(tt)-y_pred(1);
    data_in = [X y_rel(tt) y_err_last(tt) y_err_last(tt-2) y_err_last(tt-1)];
    data_in0=mapminmax('apply',data_in',inputps_err0);%数据归一化
    [y_err_last_sim,data_num0,pre_h_state0,cell_state0]=LSTM_OUT(data_num0,data_in0,weight_input_x0,...
                weight_input_h0,...
                weight_inputgate_x0,...
                weight_inputgate_c0,...
                weight_forgetgate_x0,...
                weight_forgetgate_c0,...
                weight_outputgate_x0,...
                weight_outputgate_c0,...
                bias_gate0,...
                bias_input_gate0,...
                bias_forget_gate0,...
                bias_output_gate0,...
                cell_state0,...
                input_gate0,forget_gate0,...
                output_gate0,...
                pre_h_state0,cell_num0,weight_preh_h0);
            %反归一化
y_err_last(tt)=mapminmax('reverse',y_err_last_sim,outputps_err0);%网络预测数据


% % warm start 一些LSTM一开始不准的结果
 if tt<50||y_err_last(tt)>outputps_err0.xmax||y_err_last(tt)<outputps_err0.xmin
     tt_warm_start=[tt_warm_start tt];
          y_err_last(tt) = y_rel(tt)-y_pred(1);
 end
%  if t>0%mod(t,18)<12&&mod(t,18)>6
% %             y_err_last(tt) = y_rel(tt)-y_pred(1);
%                  y_err(tt)= y_err_last(tt);
%  else
% %  % warm start 一些LSTM一开始不准的结果
  y_err_last(tt) = y_rel(tt)-y_pred(1);
              y_err(tt)= y_err_last(tt);
%  end
else
       y_err_last(tt) = y_rel(tt)-y_pred(1);
                 y_err(tt)= y_err_last(tt);
end
  
    
%     %滚动优化  梯度优化方法

                deta_u1=2*yite*Ro11*(jacobian_matrix1*(Ref_Hp-YYY)')/(1+2*yite*Ro21);
        deta_u2=2*yite*Ro12*(jacobian_matrix2*(Ref_Hp-YYY)')/(1+2*yite*Ro22);
            deta_u3=2*yite*Ro13*(jacobian_matrix3*(Ref_Hp-YYY)')/(1+2*yite*Ro23);
    wait=1;
    
     
        u1_2=u1_1;
    u1_1 = u1_1+deta_u1(1:Hc);
        u2_2=u2_1;
    u2_1 = u2_1+deta_u2(1:Hc);
        u3_2=u3_1;
    u3_1 = u3_1+deta_u3(1:Hc);
    if tt==1
        y_2=5;
    else
        y_2 =  y_rel(tt-1);
    end
    y_1=y_rel(tt);
    
    deta_u_all(:,tt)=[deta_u1(1:Hc) deta_u2(1:Hc) deta_u3(1:Hc)];
    u1_1_all=[u1_1_all u1_1];
    u1_2_all=[u1_2_all u2_1];
    u1_3_all=[u1_3_all u3_1];
    
      data_num = data_num+1;                      

end

    %------------------------------------------------------
        %-----------------算EC、NOx
    NOx_x1=[input_test_NOx(t,1:end-1) y_rel(tt)];
    NOx_x1=(NOx_x1'-inputps_NOx.min)./(inputps_NOx.max-inputps_NOx.min);
NOx_rel_temp= FNN_out(NOx_x1,Center_NOx,Width_NOx,W_NOx);
         %网络输出反归一化
         NOx_rel(t)=(outputps_NOx.max-outputps_NOx.min).*NOx_rel_temp+outputps_NOx.min;

EC_x2=[input_test_CE(t,1:end-1) y_rel(tt)];
  EC_x2=(EC_x2'-inputps_CE.min)./(inputps_CE.max-inputps_CE.min);
EC_rel_temp= FNN_out(EC_x2,Center_CE,Width_CE,W_CE);
EC_rel(t)=(outputps_CE.max-outputps_CE.min).*EC_rel_temp+outputps_CE.min;

% --------------------在线更新运行指标模型------------------
    %%%%%%%%%%%在线更新
        %     NOx增量更新NOxout
         ytest_NOx(t)=(output_test_NOx(t)-outputps_NOx.min)./(outputps_NOx.max-outputps_NOx.min);%数据归一化
%         test_xx1=[test_xx1 x1];
        test_xx11=[test_xx11 input_test_NOx(t,:)'];
%         test_yo1=[test_yo1 ytest_NOx(t)];
        test_yo11=[test_yo11 output_test_NOx(t)'];
        window_size1=10;
        if t>window_size1
%             test_xx1=test_xx1(:,end-window_size1+1:end);
%             test_yo1=test_yo1(end-window_size1+1:end);
                        test_xx11=test_xx11(:,end-window_size1+1:end);
            test_yo11=test_yo11(end-window_size1+1:end);
        end
        % 更新最大最小值
        inputps_NOx0.min=min(inputps_NOx.min,input_test_NOx(t,:)');
        inputps_NOx0.max=max(inputps_NOx.max,input_test_NOx(t,:)');
        outputps_NOx0.min=min(outputps_NOx.min,output_test_NOx(t)');
        outputps_NOx0.max=max(outputps_NOx.max,output_test_NOx(t)');
         % 更新最大最小值
                  

 if mod(t,18)==0
      % 更新最大最小值
     inputps_NOx.min = inputps_NOx0.min;
     inputps_NOx.max = inputps_NOx0.max;
     outputps_NOx.min = outputps_NOx0.min;
      outputps_NOx.max = outputps_NOx0.max;
test_xx_NOx=(test_xx11-inputps_NOx.min)./(inputps_NOx.max-inputps_NOx.min);
test_yy_NOx=(test_yo11-outputps_NOx.min)./(outputps_NOx.max-outputps_NOx.min);
[Center_NOx,Width_NOx,W_NOx] = AFNN_update_NOx(test_xx_NOx,test_yy_NOx,Center_NOx,Width_NOx,W_NOx);
 % 更新最大最小值
% [Center_NOx,Width_NOx,W_NOx] = AFNN_update(test_xx1,test_yo1,Center_NOx,Width_NOx,W_NOx);
 end

 %     EC增量更新
 ytest_EC(t)=(output_test_CE(t)-outputps_CE.min)./(outputps_CE.max-outputps_CE.min);%数据归一化
%         test_xx2=[test_xx2 x2];
%         test_yo2=[test_yo2 ytest_EC(t)];
                test_xx22=[test_xx22 input_test_CE(t,:)'];
        test_yo22=[test_yo22 output_test_CE(t)'];
        window_size2=10;
        if t>window_size2
%             test_xx2=test_xx2(:,end-window_size2+1:end);
%             test_yo2=test_yo2(end-window_size2+1:end);
                        test_xx22=test_xx22(:,end-window_size2+1:end);
            test_yo22=test_yo22(end-window_size2+1:end);
        end
        % 更新最大最小值
        inputps_CE0.min=min(inputps_CE.min,input_test_CE(t,:)');
        inputps_CE0.max=max(inputps_CE.max,input_test_CE(t,:)');
        outputps_CE0.min=min(outputps_CE.min,output_test_CE(t)');
        outputps_CE0.max=max(outputps_CE.max,output_test_CE(t)');
         % 更新最大最小值


      if mod(t,18)==0
      % 更新最大最小值
     inputps_CE.min = inputps_CE0.min;
     inputps_CE.max = inputps_CE0.max;
     outputps_CE.min = outputps_CE0.min;
      outputps_CE.max = outputps_CE0.max;
test_xx_CE=(test_xx22-inputps_CE.min)./(inputps_CE.max-inputps_CE.min);
test_yy_CE=(test_yo22-outputps_CE.min)./(outputps_CE.max-outputps_CE.min);
  [Center_CE,Width_CE,W_CE] = AFNN_update_CE(test_xx_CE,test_yy_CE,Center_CE,Width_CE,W_CE); 
  
  end
   

end

ErrHisOc=BestOxygen-y_rel(1:times_interval:end);
Num_Oc=length(ErrHisOc);
ISE_Oc=sumsqr(ErrHisOc)/Num_Oc  %计算烟气含氧量的ISE
IAE_Oc=sum(abs(ErrHisOc))/Num_Oc %计算烟气含氧量的IAE
maxdev_Oc=max(abs(ErrHisOc))  %计算烟气含氧量的最大误差偏差
ITAE_Oc=sum(time1.*abs(ErrHisOc))/Num_Oc  %计算烟气含氧量的IAE%计算烟气含氧量的ITAE        

figure;
plot(time1,BestOxygen,'k');
hold on
plot(time1,y_rel(1:times_interval:end),'-.r');
legend('\fontsize {12}Set-points','\fontsize {12}actual value')
xlabel('time(s)');
ylabel('Oxygen concentration(%)');
% ylim([4.9 6.1])
xlim([time1(1) time1(end)])
% 误差绘图
figure;
error_OC=BestOxygen-y_rel(1:times_interval:end);
plot(time1,error_OC,'k');
xlabel('time(s)');
ylabel('error(%)');
% ylim([4.8 6.2])
xlim([time1(1) time1(end)])

figure;
plot(time1,output_test_NOx,'k');
ave_output_test_NOx=mean(output_test_NOx)
hold on
plot(time1,NOx_rel,'-.r');
ave_NOx_rel=mean(NOx_rel)
xlabel('time(s)');
ylabel('NOx emission concentration(mg/m^3)');
xlim([time1(1) time1(end)])
legend('\fontsize {12}real value','\fontsize {12}optimal control value')
grid on;

figure;
plot(time1,output_test_CE,'k');
ave_output_test_EC=mean(output_test_CE)
hold on
plot(time1,EC_rel,'-.r');
ave_EC_rel=mean(EC_rel)
xlabel('time(s)');
ylabel('CE');
xlim([time1(1) time1(end)])
legend('\fontsize {12}real value','\fontsize {12}optimal control value')
grid on;

figure;
% plot(NOx_rel,'k');
% ave_NOx_rel=mean(NOx_rel)
% hold on
plot(time1,ActualNOx,'-.r');
ave_ActualNOx=mean(ActualNOx)
hold on
plot(time1,BestNOx,':b');
ave_BestNOx=mean(BestNOx)
xlabel('time(s)');
ylabel('NOx emission concentration(mg/m^3)');
xlim([time1(1) time1(end)])
legend('\fontsize {12}actual value','\fontsize {12}BestNOx')
grid on;
figure;
% plot(EC_rel,'k');
% ave_EC_rel=mean(EC_rel)
% hold on
plot(time1,ActualEC,'-.r');
ave_ActualEC=mean(ActualEC)
hold on
plot(time1,BestEC,':b');
ave_BestEC=mean(BestEC)
xlabel('time(s)');
ylabel('CE');
xlim([time1(1) time1(end)])
legend('\fontsize {12}actual value','\fontsize {12}BestCE')
grid on;

figure
plot(time1,u1_1_all(1:times_interval:end),'k','linewidth',1.5);
xlim([time1(1) time1(end)])
% xlabel('time(s)'),ylabel('error');
xlabel('时间(s)','fontsize',12);
ylabel('一次风流量（干燥段）','fontsize',12);
figure
plot(time1,u1_2_all(1:times_interval:end),'k','linewidth',1.5);
xlim([time1(1) time1(end)])
% xlabel('time(s)'),ylabel('error');
xlabel('时间(s)','fontsize',12);
ylabel('一次风流量（燃烧1段）','fontsize',12);
figure
plot(time1,u1_3_all(1:times_interval:end),'k','linewidth',1.5);
xlim([time1(1) time1(end)])
% xlabel('time(s)'),ylabel('error');
xlabel('时间(s)','fontsize',12);
ylabel('二次风流量','fontsize',12);

 figure;
 subplot(2,1,1)
plot(time1,u1_1_all(1:times_interval:end),'k','linewidth',1.5);
xlim([time1(1) time1(end)])
ylim([12.5 13.7])
% xlabel('时间(s)','fontsize',12);
% ylabel('一次风流量（干燥段）','fontsize',12);
xlabel('time(s)','fontsize',10.5);
ylabel('PA_d_r_y(km^3/h)','fontsize',10.5);
subplot(2,1,2)
plot(time1,u1_2_all(1:times_interval:end),'k','linewidth',1.5);
xlim([time1(1) time1(end)])
ylim([28.4 29.2])
% xlabel('时间(s)','fontsize',12);
% ylabel('一次风流量（燃烧1段）','fontsize',12);
xlabel('time(s)','fontsize',10.5);
ylabel('PA_i_c_n_r_1(km^3/h)','fontsize',10.5);

aaaa=1;

function y = mean5_3(x, m)
% x为被处理的数据
% m 为循环次数
n=length(x);
  a=x;
  for k=1: m
     b(1) = (69*a(1) +4*(a(2) +a(4)) -6*a(3) -a(5)) /70;
     b(2) = (2* (a(1) +a(5)) +27*a(2) +12*a(3) -8*a(4)) /35;
     for j=3:n-2
       b (j) = (-3*(a(j-2) +a(j+2)) +12*(a(j-1) +a(j+1)) +17*a(j)) /35;
     end
     b (n-1) = (2*(a(n) +a(n-4)) +27*a(n-1) +12*a(n-2) -8*a(n-3)) /35;
     b (n) = (69*a(n) +4* (a(n-1) +a(n-3)) -6*a(n-2) -a(n-4)) /70;
     a=b;
  end
  y =a;
end

