%%Getting classifying scores for supervised learning

%%creating a model for each class using + and - examples and vote for the
%%test image if it has that particular class or not

%%class i...use the trainset to create a model i...use the model for each
%%test image in the test set to determine whether it has class i or
%%not...this will provide 1 set of answers

%%To evaluate results go over all the test images
function [predicted_labels,prob_estimates]=classification(train_class,train_samples,test_class,test_samples)
    
    num_class=size(train_class,2); 

    for i=1:num_class 
       
        pos=find(train_class(:,i)==1);
        neg=find(train_class(:,i)==0);
       
      
%%%%%%%%%%%%%%%%%%%Train--Uniform%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        class=[1:num_class]; 
        class(i)=[];

        pos=find(train_class(:,i)==1);
        neg=find(train_class(:,i)==0);
        
        num_neg_samples=size(pos,1);
        num_neg_per_class=floor(num_neg_samples/(num_class-1));
        
        sub_train_samples=[];
        pick=[];
        for j=1:size(class,2) 
            index=neg(find(train_class(neg,class(j))==1));
            
            if size(index,1)>=num_neg_per_class
               pick=[pick;index(1:num_neg_per_class)];
            else
                pick=[pick;index(1:end)];
            end
            pick=unique(pick);
            k=1;
            while size(pick,1)<num_neg_per_class*j 
                if size(index,1)>=num_neg_per_class+k
                    pick=[pick;index(num_neg_per_class+k)];
                    k=k+1;
                    pick=unique(pick); 
                else
                    break
                end
            end  
        end
        
        need=num_neg_samples-size(pick,1);
        
        if need>0
           ismem=ismember(neg,pick);
           loc=find(ismem==0);
           add=neg(loc);
           pick=[pick;add(1:need)];
        end
        
        sub_train_samples=train_samples(pos,:);
        sub_train_samples=[sub_train_samples;train_samples(pick,:)];

        sub_train_labels=zeros(2*size(pos,1),1);
        sub_train_labels(1:size(pos,1))=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%Test data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
        clear pos
        test_labels=zeros(size(test_samples,1),1);
        pos=find(test_class(:,i)==1);
        test_labels(pos)=1;
                      
%%%%%%%%%%%%%%%%%%%%%%%%%--Libsvm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cd PATH/libsvm-3.17/matlab/
        model = svmtrain(sub_train_labels,sub_train_samples,'-t 0 -s 1 -q'); 
%         Using the predicted model to assign a label to each test image   decision_values=prob_estimates      
        [predicted_labels{i}, accuracy, prob_estimates{i}] = svmpredict(test_labels, test_samples, model);
        
    end
end 

