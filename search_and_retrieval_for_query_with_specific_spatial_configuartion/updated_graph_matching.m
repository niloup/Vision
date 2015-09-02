%%This will create an Affinity matrix W between a given query and all the
%%images in the DB and calculates a matching score between the query image
%%and the database images.

% clc;
% clear all;
% close all;
% clear result
% matlabpool

function [result, GM_score]=updated_graph_matching(codeFolder_match, A_Query, A, ...
    regional_Query, regional_test, Region_size_Query, Region_size, ...
    Region_position_Query, Region_position, sub_ind, updated, consider_size, consider_position)

    %% Initialization
    cd(codeFolder_match);
    run init
    run compileDir

    %% Parameter initialization
    GM_score=[];
    
    result.X=[];
    result.X_SMAC=[];
    result.timing=[];
    result.myTime=[];
    result.num_node=[];
    result.score=[];

    alpha=0.3; 
    t=0;        %%This is bc it doesnt make sense to look at the edge and node similarity
    
    n2=size(A_Query,2);   %number of nodes in the query Image
    %N=size(A,2);          %number of images in the DataBase


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

    %% Compute GM score between the query and each of the images
    result=[];
        
    %%only perform on images with index apearing in index_sim
    %%this will allow us to re-arrange the ordering of the top retrieval
    %%results.
    app_query= regional_Query;
    for i=1:size(sub_ind,2) 
     
       app_test=regional_test{sub_ind(i)};
       
       
       for temp=1:size(app_test,1)
           if (sum(isnan(app_test(temp,:)))~=0)
              app_test(temp,:)=zeros(size(app_test,2),1); 
           end
       end
       
       %%This is to take care of the query images with only 1 region
       if (n2==1)          
           %%%%%MARK TO FIND THE SCORE USING GLOBAL REPRESENTATIONS           
           result(i).score=0;
           GM_score(i)=0;
       else

           Ai=A{sub_ind(i)};
           %number of nodes in image i in the database
           n1=size(Ai,1);
           
           if (n1==1)
                result(i).score=0;
                GM_score(i)=0;
           else     
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

                        %%This is bc it doesnt make sense to look at the edge and node similarity
                        if (x1==y1 && x2~=y2) || (x1~=y1 && x2==y2)
                           W(k,j)=t;
                           W(j,k)=t;
                        else

                           %%case (k==j) is considering only the node similarities...having alpha<1 puts a higher weight on node similarities.
                           if (k==j)
                               
                                    %%see if need to consider matching regional size
                                    %%or regional positions
                                    if (consider_size==1 && consider_position==0)   
                                       weight=-norm(Region_size_Query(x2)-Region_size{sub_ind(i)}(x1));
                                    end
                                    
                                    if  (consider_size==0 && consider_position==1)
                                        weight=-norm(Region_position_Query(x2,:)-Region_position{sub_ind(i)}(x1,:));     
                                    end
                                    
                                    if  (consider_size==1 && consider_position==1)
                                         weight=-norm(Region_size_Query(x2)-Region_size{sub_ind(i)}(x1))...
                                            -norm(Region_position_Query(x2,:)-Region_position{sub_ind(i)}(x1,:));
                                        
                                    end
                                    
                                    if  (consider_size==0 && consider_position==0)
                                        weight=0;  
                                    end
                               
                               
                                if (updated==0)
                                    W(k,j)=exp(-norm(app_test(x1,:)-app_query(x2,:))/alpha + weight );
                                else
                                    W(k,j)=exp(-abs(app_test(x1)~=app_query(x2))/alpha +weight );
                                end

                           else
                                  W(k,j)=exp(-norm(double(Ai(x1,y1)-A_Query(x2,y2))));
                                  W(j,k)=exp(-norm(double(Ai(x1,y1)-A_Query(x2,y2))));    
                           end
                        end

                    end

               end



                E=ones(n1,n2);  
                tic;       
                [result(i).X,result(i).X_SMAC,result(i).timing]=compute_graph_matching_SMAC(W,E,options);
                temp=result(i).X; 
                score=temp(:)'*W*temp(:)/(temp(:)'*temp(:));
                ElapsedTime=toc;

                result(i).score=score;
                result(i).myTime=ElapsedTime;
                result(i).num_node=n1;

                GM_score(i)=score;
           end
 
       end 
    end
end
