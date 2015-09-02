%%code for updating the graphical structures
%%%Note:::representing each node by the avergae of the updated representationshistogram

%%map each node to the community with the highest likelihood to belong to
%%(largest value bin in the PixNet representation)

%%load the PixNet features...Assign each node to the community index with highest value bin
hist_test_regional=dlmread('PATH\hist_test_node_level.txt','');  
loc=dlmread('PATH/loc_test.txt','');
sup_loc=dlmread('PATH/sup_loc_test.txt','');
border=dlmread('PATH/border_test.txt','');


hist_test_regional=dlmread('PATH/hist_test_node_level.txt',''); 
loc=dlmread('PATH/loc_test.txt','');
sup_loc=dlmread('PATH/sup_loc_test.txt','');
border=dlmread('PATH/border_test.txt','');


loadPATH/Region_size_GM.mat
load PATH/Region_position_GM.mat




%%find the max and min distnace between the regional hists to determin a
%%reasonable value for alpha
min_D=zeros(1,size(hist_test_regional,1));
max_D=zeros(1,size(hist_test_regional,1));
min2_D=zeros(1,size(hist_test_regional,1));
avg_D=zeros(1,size(hist_test_regional,1));
parfor i=1:size(hist_test_regional,1)
   sample=hist_test_regional(i,:);
   D=single(pdist2(sample,hist_test_regional)); 
   min_D(i)=min(D);
   
   [x y]=sort(D);
   min2_D(i)=x(2);
   
   max_D(i)=max(D);
   avg_D(i)=sum(D)/size(D,2);
end


min_dist=min(min_D);  
min2_dist=min(min2_D);
max_dist=max(max_D);   
avg_dist=sum(avg_D)/size(avg_D,2);  
alpha=(avg_dist-min_dist)/2+min_dist;  

%%if two adjacent nodes belong to the similar communities merge them by reducing
%%the number of the nodes and connecting all the adjacent nodes of the two
%%nodes to one of the nodes and eliminating the other one

%going through all the images 
load PATH/PriA.mat
load PATH/q600_m0.7/Border/PriA.mat

DB_size=4952;
mark={};
for i=1:DB_size
    A=PriA{1,i};
    num_nodes=size(A,1);
    %%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %%find those nodes of A that have dist(hist_representations)<alpha
    for j=1:num_nodes
       mark{i}{j}=[];
       for k=j+1:num_nodes
            node1=loc(i,j);
%             node2=loc(i,k);        
            if norm(hist_test_regional(loc(i,1)+j-1,:)-hist_test_regional(loc(i,1)+k-1,:))<alpha  %%check if those regions are similar
                if (sum(border(node1,:)==k)~=0)  %check if node1 and node2 (region j and region k of image i) are neighbors
                    mark{i}{j}=[mark{i}{j}, j, k] ;
                    mark{i}{j}=unique(mark{i}{j});
                end
            end
            
       end
    end
end
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%Redefine the adjacency matrix%%%%%%%%%%%%
   
%%combine the marked regions with common nodes within
parfor i=1:DB_size
   counter=size(mark{i},2);
   j=1;
   while(j<=counter)
       k=j+1;
       while(k<=counter)
            temp=mark{i}{j};
            temp=[temp mark{i}{k}];
            [n,bin]=histc(temp,unique(temp));  % n : how many times each of the unique values repeated   
            if sum(n>1)>=1
               mark{i}{j}=unique([mark{i}{j} mark{i}{k}]);
               mark{i}{k}=[]; 
            end  
            k=k+1;
        end
        j=j+1;
       
   end
   
end
   

%%remove the empty cells
for i=1:DB_size
   counter=size(PriA{i},1);
   j=1;
   while(j<=counter)
        if (size(mark{i}{j},2)==0) 
           mark{i}(j)=[];
            counter=counter-1;
            j=j-1;
        end 
         j=j+1;
   end    
end



%%create a cell array for the updated graphs--each cell represents the
%%nodes numbers in the original graph that are being merged as one node in
%%the updated graph
for i=1:DB_size
   all_nodes=1:size(PriA{i},1);  
   merging_nodes=cell2mat(mark{i}(1:end));  %%this contains all the nodes that will be merged
   
%    merging_nodes=[];
%    for j=1:size(mark{i},2)
%         merging_nodes=[merging_nodes;mark{i}{j}];
%    end

    graph{i}=mark{i};

    k=size(mark{i},2);
    for j=1:size(PriA{i},1)
        if size(find(merging_nodes==all_nodes(j)),2)==0
            k=k+1;
            graph{i}{k}=all_nodes(j);
        end
    end
    
end



%%average number of nodes in the updated graphs
total_num_nodes_UGM=0;
total_num_nodes_GM=0;
for i=1:DB_size
    total_num_nodes_UGM=total_num_nodes_UGM+size(graph{i},2);
    total_num_nodes_GM=total_num_nodes_GM+size(PriA{i},2);
end
avg_num_nodes_UGM=total_num_nodes_UGM/DB_size;   %%15 nodes in the updates graph--%14.742 in UGM-hist
avg_num_nodes_GM=total_num_nodes_GM/DB_size;     %%22 nodes in the updates graph


%%finding the normalized size and position of each of the nodes in the
%%updated graphs

for i=1:DB_size
    for j=1:size(graph{i},2)
         Region_size_UGM{i}(j)=sum(Region_size{i}(graph{i}{j}))/size(graph{i}{j},2);
         Region_position_UGM{i}(j,1)=sum(Region_position{i}(graph{i}{j},1))/size(graph{i}{j},2);   
         Region_position_UGM{i}(j,2)=sum(Region_position{i}(graph{i}{j},2))/size(graph{i}{j},2);    
    end
end

save('Region_size_UGM_hist.mat','Region_size_UGM');
save('Region_position_UGM_hist.mat','Region_position_UGM');


%%associating each of the cells (super nodes) for each image to the average
%%of the regional community histogram representations
sub_pix_new=cell(1,DB_size);
for i=1:DB_size
   
   for j=1:size(graph{i},2)
        temp=[];
        counter=0;
        for jj=1:size(graph{i}{j},2)
            temp=[temp;hist_test_regional(loc(i,1)+graph{i}{j}(jj)-1,:)];
            counter=counter+1;
        end
        sub_pix_new{i}(j,:)=sum(temp,1)/counter;
   end
   
end


%%for each images check if the size of the mark is not zero
%%if not, merge the nodes together
PriA_new=cell(1,DB_size);
for i=1:DB_size
   
    if (size(mark{i},2)~=0)  %if there is at least two nodes that are being merged   %%(size(mark{i}{1},2)~=0) 
       dim=size(graph{i},2);
       A=zeros(dim,dim);
       
       %%for each of the mark arrays check if there is an overlap
       %%if there is, connect between the two nodes
       for j=1:size(graph{i},2)
           
           for jj=1:size(graph{i}{j},2) %%counter within the mark for cell j
               
               for k=j+1:size(graph{i},2)
                   for kk=1:size(graph{i}{k},2)  %%counter within the mark for cell k
                       
                       if PriA{i}(graph{i}{j}(jj), graph{i}{k}(kk))~=0
                            A(j,k)=1; 
                            A(k,j)=1;
                            break;
                        end
                   end
               end
               
           end
           
                    
       end
      
       PriA_new{i}=A;

   else
       
       %%use the old graphical structure
       num_nodes=size(PriA{i},1);
       PriA_new{i}=PriA{i};
       sub_pix_new{i}=hist_test_regional(loc(i,1):loc(i,1)+num_nodes-1,:);
       
   end
     
end

save('PriA_new_hist.mat','PriA_new','-v7.3')
save('sub_pix_new_hist.mat','sub_pix_new');


