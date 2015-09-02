%%%(a) Given a set of training images learn a model for each class.
%%%use that model to classify each of the test images and the query image

toolbox='PATH\vlfeat-0.9.16\toolbox\';
addpath(toolbox)
run('vl_setup') 

%%--train data
load baseline-train-hists.mat

train_samples=double(hists');
clear hists

train_samples= vl_homkermap(train_samples',1, 'kchi2', 'gamma', 0.5) ;
train_samples=double(train_samples');

load train/class.mat
train_class=class;
clear class

%%--test data
load baseline-test-hists.mat
      
test_samples=double(hists');
clear hists

test_samples= vl_homkermap(test_samples', 1, 'kchi2', 'gamma', 0.5) ;
test_samples=double(test_samples');

load test/class.mat
test_class=class;
clear class

%%--query data
query_folder='query';
query_list=dir(fullfile(query_folder, '*.jpg'))';


data_folder='JPEGImages\';
data_file=dir(fullfile(data_folder, '*.jpg'))';

%%find the j that corresponds to each query images
query_index=[];
parfor i=1:size(query_list,2)
    query_name=query_list(1,i).name;
    
    for j=1:size(data_file,2)
        data_name=data_file(1,j).name;
        if strcmp(query_name,data_name)==1
           query_index(i)=j; 
           break;
        end                
    end
    
end

save('query_index.mat','query_index');

query_samples=test_samples(query_index);
query_class=test_class(query_index);

%% -------------------------------------------------------------------------
%% ----------------------------------------------------------------------

%%--learn a model based on training data and classify each test image 
cd Retrieval
[predicted_labels,prob_estimates]=classification(train_class,train_samples,test_class,test_samples)


%---associate each image with labels that are one with the top highest probability estimates
predicted_labels=cell2mat(predicted_labels);
prob_estimates=cell2mat(prob_estimates);

sim_index=cell(size(query_index,2),1);

parfor i=1:size(query_index,2)
 
    %get the labels associated with the query image
    [value index]=sort(prob_estimates(query_index(i),:),'descend');
    est_query_label=[index(1) index(2)];
    index=[];
%     estimated_query_label=predicted_query_labels{i};
    
    %go through each test image, find their labels
    %find the index of the ones with the same label as our estimated query label 
    for j=1:size(test_class,1)
        [value index]=sort(prob_estimates(j,:),'descend');
          est_test_label=[index(1) index(2)];
           index=[];
        for k=1:size(est_query_label)
            if sum(est_test_label==est_query_label(k))~=0
               sim_index{i}=[sim_index{i},j];
               break;
            end
        end
        
    end
       
end


%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------

%%%(b1) Rank the retrieved images upto depth N using their correponding
%%%predicted scores OR hellinger distnace as a similarity metric

all_index=1:size(test_class,1);

val=zeros(size(query_index,2),size(test_samples,1));
ind=zeros(size(query_index,2),size(test_samples,1));

parfor i=1:size(query_index,2)  
    query_feature=test_samples(query_index(i),:);   
    
    val1=[];val2=[];ind1=[];ind2=[];
    
    %sort images that are classified as the same class as the query
    sub_pos_test_samples=test_samples(sim_index{i},:);
    D=pdist2(query_feature,sub_pos_test_samples);
    pos_index=sim_index{i};
    [val1,ind1]=sort(D,'ascend');
    
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(ismember( all_index ,sim_index{i})==0);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(query_feature,sub_neg_test_samples);
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_HD.mat','ind','val');
 

%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------

%%%(b2) Re-rank them using GM score

load Regional_phow_test.mat
regional_test=hist_hard_Q;
clear hist_hard_Q

load PriA.mat
A=PriA;
clear PriA;

updated=0;   %%performing the graph matching using initial graphical representations
all_index=1:size(test_class,1);

val=zeros(size(query_index,2),size(test_class,1));
ind=zeros(size(query_index,2),size(test_class,1));

all_index=1:size(test_class,1);

parfor i=1:size(query_index,2)  
    i
    query_feature=test_samples(query_index(i),:);   
    
     val1=[];val2=[];ind1=[];ind2=[];
       
    %get the graph matching score between the query and images classified
    %to have the same label as the query image    
    A_Query=A{query_index(i)};
    regional_Query=regional_test{query_index(i)};
    sub_ind=sim_index{i};
   
    codeFolder_match='PATH/graph_matching_SMAC/';
    [result GM_score{i}]=updated_graph_matching(codeFolder_match, A_Query, A, regional_Query, regional_test, sub_ind, updated);
    
    GM_score_normalized=GM_score{i}/max(GM_score{i});
    

    sub_pos_test_samples=test_samples(sim_index{i},:);
    D=pdist2(sqrt(query_feature),sqrt(sub_pos_test_samples));
    vis_sim=(exp(1)*ones(size(D))).^(-1.*D);
    vis_sim=vis_sim/max(vis_sim);
    
    tot_sim=(GM_score_normalized+vis_sim)*0.5;
    
    pos_index=sim_index{i};
    [val1,ind1]=sort(tot_sim,'descend');
       
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(~ismember( all_index ,sim_index{i})==1);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(sqrt(query_feature),sqrt(sub_neg_test_samples));
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_GM_mixed.mat','val','ind');

%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------
%%%(b3)Re-rank them using updated rule and GM score
load PriA_new.mat
A=PriA_new;
clear PriA_new


%%%if representing each updated node with the community that has the highest membership likelihood
load sub_pix_new.mat
regional_test=sub_pix_new;
clear sub_pix_new

updated=1;   %%performing the graph matching using updated graphical representations
all_index=1:size(test_class,1);
codeFolder_match='PATH/graph_matching_SMAC/';
cd(codeFolder_match)

all_index=1:size(test_class,1);

val=zeros(size(query_index,2),size(test_samples,1));
ind=zeros(size(query_index,2),size(test_samples,1));

GM_score={};
parfor i=1:size(query_index,2)  
    i
    query_feature=test_samples(query_index(i),:);   
    
    val1=[];val2=[];ind1=[];ind2=[];
    
    sub_ind=sim_index{i};
    
    %get the graph matching score between the query and images classified
    %to have the same label as the query image 
    A_Query=A{query_index(i)};
    regional_Query=regional_test{query_index(i)};
    cd /media/Universe/school/Retrieval/
    [result GM_score{i}]=updated_graph_matching(codeFolder_match, A_Query, A, regional_Query, regional_test, sub_ind, updated);
        
    GM_score_normalized=GM_score{i}/max(GM_score{i});
    
    sub_pos_test_samples=test_samples(sim_index{i},:);
    D=pdist2(sqrt(query_feature),sqrt(sub_pos_test_samples));
    vis_sim=(exp(1)*ones(size(D))).^(-1.*D);
    vis_sim=vis_sim/max(vis_sim);
    
    tot_sim=(GM_score_normalized+vis_sim)*0.5;
    
    pos_index=sim_index{i};
    [val1,ind1]=sort(tot_sim,'descend');
    
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(~ismember( all_index ,sim_index{i})==1);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(sqrt(query_feature),sqrt(sub_neg_test_samples));
    
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_UGM_mixed.mat','val','ind');

%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------
%% ----------------------------------------------------------------------
%%%Learn the model (a) using PixNet features and repeat the entire process.
test_samples=dlmread('PATH/hist_test_sim.txt','');

all_index=1:size(test_class,1);

val=zeros(size(query_index,2),size(test_samples,1));
ind=zeros(size(query_index,2),size(test_samples,1));

parfor i=1:size(query_index,2)  
    query_feature=test_samples(query_index(i),:);   
    
    val1=[];val2=[];ind1=[];ind2=[];
    
    %sort images that are classified as the same class as the query
    sub_pos_test_samples=test_samples(sim_index{i},:);
    D=pdist2(query_feature,sub_pos_test_samples);
    pos_index=sim_index{i};
    [val1,ind1]=sort(D,'ascend');
    
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(ismember( all_index ,sim_index{i})==0);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(query_feature,sub_neg_test_samples);
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_PixNet.mat','ind','val','sim_index','prob_estimates','predicted_labels');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%(b4) Re-rank them using GM score and the location and position encoding

load PATH\Regional_phow_test.mat
regional_test=hist_hard_Q;
clear hist_hard_Q

load PATH\PriA.mat
A=PriA;
clear PriA;

load PATH/Region_position_GM.mat
load PATH/Region_size_GM.mat


updated=0;   %%performing the graph matching using initial graphical representations
consider_size=1;
consider_position=1;

val=zeros(size(query_index,2),size(test_class,1));
ind=zeros(size(query_index,2),size(test_class,1));

all_index=1:size(test_class,1);

GM_score={};
parfor i=1:size(query_index,2)
    i
    query_feature=test_samples(query_index(i),:);   
    
     val1=[];val2=[];ind1=[];ind2=[];
    
       
    %get the graph matching score between the query and images classified
    %to have the same label as the query image
    
    A_Query=A{query_index(i)};
    regional_Query=regional_test{query_index(i)};
    Region_size_Query=Region_size{query_index(i)};
    Region_position_Query=Region_position{query_index(i)};
    
    
    sub_ind=sim_index{i};
    
    codeFolder_match='PATH/graph_matching_SMAC/';
    
    [result score]=updated_graph_matching(codeFolder_match, A_Query, A, regional_Query, regional_test, ...
        Region_size_Query, Region_size, Region_position_Query, Region_position,...
        sub_ind, updated, consider_size, consider_position);
    
    GM_score{i}=score;
         
    GM_score_normalized=GM_score{i}/max(GM_score{i});
    

    sub_pos_test_samples=test_samples(sim_index{i},:);
    D=pdist2(sqrt(query_feature),sqrt(sub_pos_test_samples));
    vis_sim=(exp(1)*ones(size(D))).^(-1.*D);
    vis_sim=vis_sim/max(vis_sim);
    
    tot_sim=(GM_score_normalized+vis_sim)*0.5;
    
    pos_index=sim_index{i};
    [val1,ind1]=sort(tot_sim,'descend');
        
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(~ismember( all_index ,sim_index{i})==1);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(sqrt(query_feature),sqrt(sub_neg_test_samples));
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_GM_mixed_pos_size.mat','val','ind');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%(b3)Re-rank them using updated rule and GM score + considering size and
%%%position

load F:\school\retrieval\VOC\PriA_new_q600.mat
A=PriA_new;
clear PriA_new


%%%if representing each updated node with the community that has the
%%%highest membership likelihood
load PATH\sub_pix_new.mat
regional_test=sub_pix_new;
clear sub_pix_new

updated=1;   %%performing the graph matching using updated graphical representations
all_index=1:size(test_class,1);
codeFolder_match='PATH/graph_matching_SMAC/';
cd(codeFolder_match)
run init
run compileDir


load PATH/Region_position_UGM.mat

load PATH/Region_size_UGM.mat


updated=1;   %%performing the graph matching using initial graphical representations
consider_size=0;
consider_position=0;

val=zeros(size(query_index,2),size(test_class,1));
ind=zeros(size(query_index,2),size(test_class,1));

all_index=1:size(test_class,1);

GM_score={};
parfor i=1:size(query_index,2)
    i
    query_feature=test_samples(query_index(i),:);   
    
     val1=[];val2=[];ind1=[];ind2=[];
    
       
    %get the graph matching score between the query and images classified
    %to have the same label as the query image
    
    A_Query=A{query_index(i)};
    regional_Query=regional_test{query_index(i)};
    Region_size_Query=Region_size{query_index(i)};
    Region_position_Query=Region_position{query_index(i)};
    
    
    sub_ind=sim_index{i};
    
    codeFolder_match='PATH/graph_matching_SMAC/';

     [result score]=updated_graph_matching(codeFolder_match, A_Query, A, regional_Query, regional_test, ...
         Region_size_Query, Region_size, Region_position_Query, Region_position,...        
         sub_ind, updated, consider_size, consider_position);
     
     GM_score{i}=score;
   
        
     GM_score_normalized=GM_score{i}/max(GM_score{i});

    
     sub_pos_test_samples=test_samples(sim_index{i},:);
     D=pdist2(sqrt(query_feature),sqrt(sub_pos_test_samples));
     vis_sim=(exp(1)*ones(size(D))).^(-1.*D);
     vis_sim=vis_sim/max(vis_sim);
     
     tot_sim=(GM_score_normalized+vis_sim)*0.5;
    
     pos_index=sim_index{i};
    
     [val1,ind1]=sort(tot_sim,'descend');

    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(~ismember( all_index ,sim_index{i})==1);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(sqrt(query_feature),sqrt(sub_neg_test_samples));
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_UGM_mixed.mat','val','ind');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%(b3)Re-rank them using updated rule_hist and GM score + considering size and
%%%position

load PATH\PriA_new_hist.mat
A=PriA_new;
clear PriA_new


%%%if representing each updated node with the community that has the
%%%highest membership likelihood
load PATH\sub_pix_new_hist.mat
regional_test=sub_pix_new;
clear sub_pix_new

all_index=1:size(test_class,1);

codeFolder_match='PATH/graph_matching_SMAC/';
run init
run compileDir


load PATH/Region_position_UGM_hist.mat

load PATH/Region_size_UGM_hist.mat


updated=0;   %%performing the graph matching using initial graphical representations
consider_size=1;
consider_position=1;

val=zeros(size(query_index,2),size(test_class,1));
ind=zeros(size(query_index,2),size(test_class,1));

all_index=1:size(test_class,1);


GM_score={};
parfor i=1:size(query_index,2)
    i
    query_feature=test_samples(query_index(i),:);   
    
     val1=[];val2=[];ind1=[];ind2=[];
    
       
    %get the graph matching score between the query and images classified
    %to have the same label as the query image   
    A_Query=A{query_index(i)};
    regional_Query=regional_test{query_index(i)};
    Region_size_Query=Region_size{query_index(i)};
    Region_position_Query=Region_position{query_index(i)};
    
    
    sub_ind=sim_index{i};
    
     cd /media/Universe/school/Retrieval/
     [result score]=updated_graph_matching(codeFolder_match, A_Query, A, regional_Query, regional_test, ...
          Region_size_Query, Region_size, Region_position_Query, Region_position,...        
          sub_ind, updated, consider_size, consider_position);
     
     GM_score{i}=score;
    
     GM_score_normalized=GM_score{i}/max(GM_score{i});

    
     sub_pos_test_samples=test_samples(sim_index{i},:);
     D=pdist2(sqrt(query_feature),sqrt(sub_pos_test_samples));
     vis_sim=(exp(1)*ones(size(D))).^(-1.*D);
     vis_sim=vis_sim/max(vis_sim);
     
     tot_sim=(GM_score_normalized+vis_sim)*0.5;
    
     pos_index=sim_index{i};
    
     [val1,ind1]=sort(tot_sim,'descend');
            
    %sort images that are not classified as the same class as the query
    %--first get the index of test images that are not in sim_index{i}    
    neg_index=find(~ismember( all_index ,sim_index{i})==1);
    sub_neg_test_samples=test_samples(neg_index,:);
    D=pdist2(sqrt(query_feature),sqrt(sub_neg_test_samples));
    [val2,ind2]=sort(D,'ascend');
        
    ind(i,:)=[pos_index(ind1) neg_index(ind2)];
    val(i,:)=[val1 val2];
    
end

save('query_UGM_hist_mixed.mat','val','ind');
