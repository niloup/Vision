%% Niloufar Pourian
%This will create an Affinity matrix W between a given query and all the
%images in the DB and calculates a matching score between the query image
%and the database images.

% clc;
% clear all;
% close all;

% cd /media/Universe/school/Segmentation/ExtractFeatures/
 
% load Border/PriA
% load hist/hist_Region
% load hist/hist_Boundry
 
% % matlabpool
% imageFolder_DB='E:\school\Retrieval\DataBases\Oxford5K\oxbuild_images\';
imageFolder_DB='/media/Universe/school/Retrieval/DataBases/Oxford5K/oxbuild_images/';
% imageFolder_DB='/cluster/data/nilou/DataBase/Oxford5k/oxbuild_images/';
fileList_DB = dir(fullfile(imageFolder_DB, '*.jpg'))';

%% Finding query, good and junk image number
count=0;
for i=1:5063
   if strcmp(fileList_DB(1,i).name,'christ_church_000179.jpg')==1
       query=i;
       count=count+1;
   elseif strcmp(fileList_DB(1,i).name,'christ_church_000366.jpg')==1
       good=i;
       count=count+1;
   elseif strcmp(fileList_DB(1,i).name,'oxford_003410.jpg')==1
       junk=i;
       count=count+1;
   end
   if count==3
       break
   end
end

%% Finding Atest
i=junk;
num_nodes=size(PriA{1,i},1);
if num_nodes == 0
  Ai=[];
else
   Ai=cell(num_nodes);
   for j=1:num_nodes
      temp=hist_Region{i,j}/(hist_Region{i,j}.^2);
      Ai{j,j}=temp/sum(temp);
      for k=j+1:num_nodes   
            if sum(hist_Boundry{1,i}{j,k}~=0)
              temp=hist_Boundry{1,i}{j,k}/sum(hist_Boundry{1,i}{j,k}.^2); 
              Ai{j,k}=temp/sum(temp); 
              Ai{k,j}=temp/sum(temp); 
            else
               Ai{j,k}=hist_Boundry{1,i}{j,k};
               Ai{k,j}=hist_Boundry{1,i}{j,k};
            end
      end
   end
end
A_test=Ai;   
n1=size(A_test,1)

%%  Finding AQuery
i=query;
num_nodes=size(PriA{1,i},1);
if num_nodes == 0
  Ai=[];
else
   Ai=cell(num_nodes);
   for j=1:num_nodes
      temp=hist_Region{i,j}/(hist_Region{i,j}.^2);
      Ai{j,j}=temp/sum(temp);
      for k=j+1:num_nodes   
          if sum(hist_Boundry{1,i}{j,k}~=0)
              temp=hist_Boundry{1,i}{j,k}/sum(hist_Boundry{1,i}{j,k}.^2); 
              Ai{j,k}=temp/sum(temp); 
              Ai{k,j}=temp/sum(temp); 
          else
               Ai{j,k}=hist_Boundry{1,i}{j,k};
               Ai{k,j}=hist_Boundry{1,i}{j,k};
          end
      end
   end
end
A_query=Ai;   
n2=size(A_query,2)


%% Get the Matching Score
% cd E:\school\GraphMatching\graph_matching_SMAC
% cd C:\Users\Niloufar\Desktop\graph_matching_SMAC
cd /media/Universe/school/GraphMatching/graph_matching_SMAC/
save('test2.mat','A_query','A_test','n1','n2','-v7.3');
%% options for graph matching (discretization, normalization)
options.constraintMode='both'; %'both' for 1-1 graph matching
options.isAffine=1;% affine constraint
options.isOrth=1;%orthonormalization before discretization
options.normalization='iterative';%bistochastic kronecker normalization
% options.normalization='none'; %we can also see results without normalization
options.discretisation=@discretisationGradAssignment; %function for discretization
options.is_discretisation_on_original_W=1; %--Note: check if 0 or 1 gives better results
% 
% %put options.is_discretisation_on_original_W=1 if you want to discretize on original W 
% %1: might be better for raw objective (based orig W), but should be worse for accuracy

%%Tuning parameters
alpha1=1;
gamma1=0.01;
alpha2=1;
gamma2=0.001;

% % run init
% % run compileDir
   
W=zeros(n1*n2,n1*n2);

for k=1:n1*n2
    for j=1:n1*n2
        x1=mod(k,n1);
        if x1==0 
           x1=n1;
        end
        y1=mod(j,n1);
        if y1==0 
           y1=n1;
        end

        x2=ceil(k/n1);
        if x2==0 
           x2=n2;
        end
        y2=ceil(j/n1);
        if y2==0 
           y2=n2;
        end
        
      if k==j
          W(k,j)=alpha1*exp(-norm(A_test{x1,y1}-A_query{x2,y2})*gamma1);
          W(j,k)=alpha1*exp(-norm(A_test{x1,y1}-A_query{x2,y2})*gamma1);     
      else    
          W(k,j)=alpha2*exp(-norm(A_test{x1,y1}-A_query{x2,y2})*gamma2);
          W(j,k)=alpha2*exp(-norm(A_test{x1,y1}-A_query{x2,y2})*gamma2);        
      end
      
    end
end

E=ones(n1,n2);
tic;
[X,X_SMAC,timing]=compute_graph_matching_SMAC(W,E,options);
temp=X; 
score=temp(:)'*W*temp(:)/(temp(:)'*temp(:))
ElapsedTime=toc

    
% 
% cd(saveFolder)
% save('scores.mat','results');
% fprintf('Done \n');