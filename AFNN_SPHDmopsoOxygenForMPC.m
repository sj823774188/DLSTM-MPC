function [bestOxygen,bestNOx,bestEC] = AFNN_SPHDmopsoOxygenForMPC(XTest_NOx,XTest_EC,Center_NOx,Width_NOx,W_NOx,inputps_NOx,outputps_NOx,Center_CE,Width_CE,W_CE,inputps_CE,outputps_CE)

% %固定随机种子,使其训练结果不变
% setdemorandstream(pi);
%% Initialize all parameter of the algorithm
maxIterations = 200;        % Maximum Number of Iterations
swarmSize = 50;            % Swarm Size
repSize = 50;              % Repository Size
nVar = 1;                   %一个决策变量
pm = 1/nVar;                % Mutation Rate
varSize = [1 nVar];   % Size of Decision Variables Matrix
mutationPara = 1;           % Scale Factor of Non-highly (p=0) and Highly (p=1) Disruptive Polynomial Mutation%ZDT1\ZDT2 for p=1;ZDT3 for p=0,
distributionIndex = 20;     % Distribution Index for Polynomial Mutation

r1Max = 1;
r1Min = 0;
r2Max = 1;
r2Min = 0;
c1Max = 2.5;
c1Min = 1.5;
c2Max = 2.5;
c2Min = 1.5;
weightMax = 0.5;
weightMin = 0.1;
varMax = 6;%烟气含氧量的上限
varMin = 5;%烟气含氧量的下限
deltaMax = (varMax - varMin)/2; % Upper Boundary of Velocity
deltaMin = -1*deltaMax;         % Lower Boundary of Velocity
% Paper [3] said that for changeVelocity 0.001 is better than -1 in paper
% [1], but in my actual test, -1 is better, and 0.001 even can't work.
changeVelocity1 = 0.001;       % Change Coefficient of Velocity when a Particle Goes beyond it's Lower Boundary
changeVelocity2 = 0.001;       % Change Coefficient of Velocity when a Particle Goes beyond it's Upper Boundary

cgma0=0.2;

% Create the population structure
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.Best.OverallConstraintViolation = [];
empty_particle.Distance = [];
empty_particle.OverallConstraintViolation = [];

swarm = repmat(empty_particle, swarmSize, 1);
rep = [];
    
for i = 1:swarmSize
    % Create the initial population and evaluate
	swarm(i).Position = unifrnd(varMin, varMax, varSize);
%     swarm(i).Cost = evaluate(costFunctionType, swarm(i).Position);
    % 适应度函数
    x1=[XTest_NOx(:,1:end-1) swarm(i).Position];
x1=(x1'-inputps_NOx.min)./(inputps_NOx.max-inputps_NOx.min);
     ym_NOx= FNN_out(x1,Center_NOx,Width_NOx,W_NOx);
    
    x2=[XTest_EC(:,1:end-1) swarm(i).Position];
x2=(x2'-inputps_CE.min)./(inputps_CE.max-inputps_CE.min);
    ym_EC= FNN_out(x2,Center_CE,Width_CE,W_CE);
    
%         %网络输出反归一化
%          ym_NOx=(outputps_NOx.max-outputps_NOx.min).*ym_NOx+outputps_NOx.min;
% ym_EC=(outputps_CE.max-outputps_CE.min).*ym_EC+outputps_CE.min;
        
    swarm(i).Cost=[-ym_EC;ym_NOx];
%     if nCons ~= 0
%         swarm(i).OverallConstraintViolation = evaluateConstraints(costFunctionType, swarm(i).Position, nCons);
%     end
    
    % Initial the speed of each particle to 0
    swarm(i).Velocity = zeros(varSize);
    
    % Add the particle to rep if it's non-dominated
    nCons=0;
    rep = leadersAdd(swarm(i), rep, repSize, nCons);

	% Initialize the memory of each particle
	swarm(i).Best.Position = swarm(i).Position;
	swarm(i).Best.Cost = swarm(i).Cost;
    swarm(i).Best.OverallConstraintViolation = swarm(i).OverallConstraintViolation;
end

% Crowding the leaders
rep = crowdingDistanceAssignment(rep);


%% MOPSO Main Loop

for it = 1:maxIterations

    CostFunction=[];
    % Compute the speed
    [leader,swarm] = computeSpeed(swarm, rep, weightMin, weightMax, r1Min, r1Max, r2Min, r2Max, ...
        c1Min, c1Max, c2Min, c2Max, deltaMin, deltaMax,it,maxIterations,varMin,varMax,CostFunction);

    % Compute the new positions for the particles
    for i = 1:swarmSize
        % Update the position of each particle
		swarm(i).Position = swarm(i).Position + swarm(i).Velocity;
		% Maintain the particle whin the search space in case they go beyond their boundaries.
        for k = 1:nVar
            if swarm(i).Position(k) < varMin(k)
                swarm(i).Position(k) = varMin(k);
                swarm(i).Velocity(k) = changeVelocity1*swarm(i).Velocity(k);
            end
            if swarm(i).Position(k) > varMax(k)
                swarm(i).Position(k) = varMax(k);
                swarm(i).Velocity(k) = changeVelocity2*swarm(i).Velocity(k);
            end
        end
    end
   
    
    % Mutate the particles
    for i = 1:swarmSize
        if mod(i, 6) == 0
            % Polynomial Mutation
            swarm(i).Position = polynomialMutation(swarm(i).Position, pm, varMin, ...
                varMax, mutationPara, distributionIndex);
        end
    end
    
    % Evaluate the new particles in new positions
    for i = 1:swarmSize
%         swarm(i).Cost = evaluate(costFunctionType, swarm(i).Position);
         % 适应度函数
    x1=[XTest_NOx(:,1:end-1) swarm(i).Position];
x1=(x1'-inputps_NOx.min)./(inputps_NOx.max-inputps_NOx.min);
     ym_NOx= FNN_out(x1,Center_NOx,Width_NOx,W_NOx);
    
    x2=[XTest_EC(:,1:end-1) swarm(i).Position];
x2=(x2'-inputps_CE.min)./(inputps_CE.max-inputps_CE.min);
    ym_EC= FNN_out(x2,Center_CE,Width_CE,W_CE);
    
%         %网络输出反归一化
%          ym_NOx=(outputps_NOx.max-outputps_NOx.min).*ym_NOx+outputps_NOx.min;
% ym_EC=(outputps_CE.max-outputps_CE.min).*ym_EC+outputps_CE.min;
        
    swarm(i).Cost=[-ym_EC;ym_NOx];   
        
%       if nCons ~= 0
%         for i = 1:swarmSize
%             swarm(i).OverallConstraintViolation = evaluateConstraints(costFunctionType, swarm(i).Position, nCons);
%         end
%     end      
%         
 % Actualize the Repository
    for i = 1:swarmSize
        rep = leadersAdd(swarm(i), rep, repSize, nCons);
    end
    
    % Actualize the memory of this particle
    for i = 1:swarmSize
        if dominanceCompare(swarm(i), swarm(i).Best, nCons) ~= -1
            swarm(i).Best.Position = swarm(i).Position;
			swarm(i).Best.Cost = swarm(i).Cost;
        end
    end
    
    % Crowding the leaders
    rep = crowdingDistanceAssignment(rep);

   

	% Show Iteration Information
	disp(['Iteration ' num2str(it) ': Number of Rep Members = ' num2str(numel(rep))]);

    end
end

% %---------------乌托邦点决策-------------------
rep_fitness=[rep.Cost];

% % %--------------------
[phi1,index1]=min(rep_fitness(1,:));
[phi2,index2]=min(rep_fitness(2,:));
Utopia_point=[phi1;phi2];
AA=rep_fitness-Utopia_point;
for i=1:size(AA,2)
    CC(i)=norm(AA(:,i),1);
end
% eita=0.79;
% for i=1:size(AA,2)
%     CC(i)=eita.*AA(1,i)+(1-eita).*AA(2,i);
% end
[optimal_point,index_optimal]=min(CC);

figure(2);
% Plotfitness2(rep_fitness_scaled,index_optimal);
Plotfitness2(rep_fitness,index_optimal);


%% 输出结果
% 最好的参数
bestOxygen = rep(index_optimal).Position;

x1=[XTest_NOx(:,1:end-1) bestOxygen];
x1=(x1'-inputps_NOx.min)./(inputps_NOx.max-inputps_NOx.min);
bestNOx= FNN_out(x1,Center_NOx,Width_NOx,W_NOx);

x2=[XTest_EC(:,1:end-1) bestOxygen];
x2=(x2'-inputps_CE.min)./(inputps_CE.max-inputps_CE.min);
bestEC= FNN_out(x2,Center_CE,Width_CE,W_CE);


