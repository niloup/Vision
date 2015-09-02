%This will create an Affinity matrix W between a given query and all the
%images in the DB and calculates a matching score between the query image
%and the database images.

% clc;
% clear all;
% close all;
% clear result
% matlabpool

saveFolder='PATH/GM_withGT/';
codeFolder_match='PATH/graph_matching_SMAC';

% load 'A200.mat';
% load 'A200_wo_squaredDivision.mat'
% load 'A200_wo_squaredDivision_notNormalized.mat'
% load 'A200_wo_normal.mat';
% load 'A200_jseg.mat';

%number of images in the DataBase
N=size(A,2);

load PATH/query_index.mat

cd(codeFolder_match);

run init
run compileDir

result.X=[];
result.X_SMAC=[];
result.timing=[];
result.myTime=[];
result.num_node=[];
result.score=[];
% score=zeros(size(query_index,2),N);


alpha=0.3; 
for m=1:size(query_index,2)
    m
    A_Query=A{1,query_index(m)};
    %number of nodes in the query Image
    n2=size(A_Query,2);
    
    for i=1:N
       
       %%This is to take care of the images with only 1 region
       if (n2==1)          
           %%%%%MARK TO FIND THE SCORE USING GLOBAL REPRESENTATIONS           
            result(m,i).score=0;
       else
           
           Ai=A{i};
           %number of nodes in image i in the database
           n1=size(Ai,1);

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

                    y2=ceil(j/n1);
                    
                    t=0;
                    
                    %%This is bc it doesnt make sense to look at the edge
                    %%and node similarity
                    if (x1==y1 && x2~=y2) || (x1~=y1 && x2==y2)
                       W(k,j)=t;
                       W(j,k)=t;
                    else
                       %%case k==j is considering only the node
                       %%similarities...having alpha<1 puts a higher weight
                       %%on node similarities.
                       if k==j
                          W(k,j)=exp(-norm(Ai{x1,y1}-A_Query{x2,y2})/alpha);
                       else
                              W(k,j)=exp(-norm(Ai{x1,y1}-A_Query{x2,y2}));
                              W(j,k)=exp(-norm(Ai{x1,y1}-A_Query{x2,y2}));    
                       end
                    end

                end
                 
           end

            
            E=ones(n1,n2);
            %% options for graph matching (discretization, normalization)
            options.constraintMode='both'; %'both' for 1-1 graph matching
            options.isAffine=1;% affine constraint
            options.isOrth=1;%orthonormalization before discretization
            options.normalization='iterative';%bistochastic kronecker normalization
%             options.normalization='none'; %we can also see results without normalization
            options.discretisation=@discretisationGradAssignment; %function for discretization
            options.is_discretisation_on_original_W=1; %--Note: check if 0 or 1 gives better results
            % 
            % %put options.is_discretisation_on_original_W=1 if you want to discretize on original W 
            % %1: might be better for raw objective (based orig W), but should be worse for accuracy
            %%
            tic;
            [result(m,i).X,result(m,i).X_SMAC,result(m,i).timing]=compute_graph_matching_SMAC(W,E,options);
            temp=result(m,i).X; 
            score1=temp(:)'*W*temp(:)/(temp(:)'*temp(:));
             
            result(m,i).score=score1;
            
            ElapsedTime=toc;
            result(m,i).myTime=ElapsedTime;
            result(m,i).num_node=n1;
        end
    end       
end
