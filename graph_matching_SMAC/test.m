%% Written by: Niloufar Pourian
%This function will compute a matching score between two graphs

%{
% Timothee Cour, 4 february 2009: fixed a bug which affected discretization in the case n1~=n2
% This software is made publicly for research use only.
% It may be modified and redistributed under the terms of the GNU General Public License.

nbTrials=100;
nbErrors=zeros(nbTrials,1);
for i=1:nbTrials
nbErrors(i)=demo_graph_matching_SMAC();
end
disp2('mean(nbErrors)');
%}

clc;
% clear all;
% close all;

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
% 
%% compute a synthetic graph matching example
% n1=5; %number of nodes in G1
% n2=3; %number of nodes in G2
% n3=3; %number of nodes in G3

% A1=zeros(n1,n1);
% A1=[10 1 1 1 0;1 3 1 0 0;1 1 1 0 0;1 0 0 1 1;0 0 0 1 1];
% A2=zeros(n2,n2);
% A3=[8 1 1;1 4 1;1 1 1];
% n2=size(A2,1);
% 
% A3=zeros(n3,n3);
% A3=[5 1 1;1 0 0;1 0 0];

A1=[8 4 5 0 1 0 0 0;4 5.5 0 5 0 0 0 0;5 0 8.5 7 0 0 0 6;0 5 7 3 0 0 0 0;...
1 0 0 0 3 4 3 0;0 0 0 0 4 1 0 0;0 0 0 0 3 0 10 0;0 0 6 0 0 0 0 10];

A3=[9 4.5 3 0;4.5 5 0 4;3 0 8 6;0 4 6 3];

% A1=[10 1 0 1 0;1 3 0 0 0;0 0 1 1 1;1 0 1 1 1;0 0 1 1 1]
% % 
% A3=[8 1 1;1 4 0;10 0 1]

n1=size(A1,1);
n3=size(A3,1);

% e=-3;  %this is for the ones with no edge between them
% k=1;    %this is representing the nodes
% A1=[1 1 5 e 3 e e e;1 1 e 6 e e e e;5 e 1 1 e e e 1;e 6 1 1 e e e e;...
%     1 e e e 1 1 1 e;e e e e 1 1 e e;e e e e 1 e 1 e;e e 1 e e e e 1];
% n1=size(A1,1);
% 
% A3=[1 1 5 e;1 1 e 4;5 e 1 1;e 4 1 1];
% n3=size(A3,1);

% A1=[1 1 5 0 1 0 0 0;1 1 0 3 0 0 0 0;5 0 1 1 0 0 0 1;0 3 1 1 0 0 0 0;...
%     1 0 0 0 1 1 1 0;0 0 0 0 1 1 0 0;0 0 0 0 1 0 1 0;0 0 1 0 0 0 0 1];
% 
% A3=[1 1 5 0;1 1 0 3;5 0 1 1;0 3 1 1];

%---Creating Affinity Matrix Between G1 nad G3--(ElapsedTime =0.6799)

W13=zeros(n1*n3,n1*n3);

for i=1:n1*n3
    for j=1:n1*n3
        x1=mod(i,n1);
        if x1==0 
           x1=n1;
        end
        y1=mod(j,n1);
        if y1==0 
           y1=n1;
        end
      
        x3=ceil(i/n1);
        
        y3=ceil(j/n1);
       
        t=0;
        if (x1==y1 && x3~=y3) || (x1~=y1 && x3==y3)
            W13(i,j)=t;
            W13(j,i)=t;
        else
            %%might want to consider the case if the edges are zero..maytbe give a diff weight ...if A1(x1,y1)=
            W13(i,j)=exp(-norm(A1(x1,y1)-A3(x3,y3)));
            W13(j,i)=exp(-norm(A1(x1,y1)-A3(x3,y3)));        
        end
               
%       W13(i,j)=exp(-norm(A1(x1,y1)-A3(x3,y3)));
%       W13(j,i)=exp(-norm(A1(x1,y1)-A3(x3,y3)));        
        
    end
end

E13=ones(n1,n3);
% tic;
[X13,X_SMAC3,timing]=compute_graph_matching_SMAC(W13,E13,options);

% score=X13(:)'*triu(W13)*X13(:)/(X13(:)'*X13(:))
score13=X13(:)'*W13*X13(:)/(X13(:)'*X13(:)) %--not sure if this is right
X13
% ElapsedTime=toc

%---Creating Affinity Matrix Between G1 nad G2--(ElapsedTime = 0.0559)
% W=zeros(n1*n2,n1*n2);
% for i=1:n1*n2
%     for j=1:n1*n2
%         x1=mod(i,n1);
%         if x1==0 
%            x1=n1;
%         end
%         y1=mod(j,n1);
%         if y1==0 
%            y1=n1;
%         end
%         
%         x2=ceil(i/n1);
%         if x2==0 
%            x2=n2;
%         end
%         y2=ceil(j/n1);
%         if y2==0 
%            y2=n2;
%         end
%         
% %         disp([i x1 x2 j y1 y2])
%                
%       W(i,j)=exp(-norm(A1(x1,y1)-A2(x2,y2)));
%       W(j,i)=exp(-norm(A1(x1,y1)-A2(x2,y2)));        
%         
%     end
% end
% 
% E12=ones(n1,n2);
% tic;
% [X12,X_SMAC,timing]=compute_graph_matching_SMAC(W,E12,options);
% 
% score12=X12(:)'*W*X12(:)/(X12(:)'*X12(:)) %Here score12>score13 which makes sense!
% 
% ElapsedTime=toc

%% results evaluation
% if n1>n2
%     dim=1;
% else
%     dim=2;
% end
% [ignore,target_ind]=max(target,[],dim);
% [ignore,X12_ind]=max(X12,[],dim);
% nbErrors=sum(X12_ind~=target_ind)/length(target_ind);
% 
% score=computeObjectiveValue(W,X12(E12>0));

%timing for SMAC (excluding discretization, which is not optimized)
% timing
