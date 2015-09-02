%% %%%%%%%%%%%%% computing the AP %%%%%%%%%%%%%%%%%%%%%%%%
% DB_size=TO BE SET;
% num_class=TO BE SET;

sim=zeros(size(query_index,2),DB_size);
for m=1:size(query_index,2)
   for i=1:DB_size
      sim(m,i)=result(m,i).score; 
   end
end

for i=1:size(query_index,2)
    query_class=find(class(query_index(i),:)==1);
    temp=find(class(:,query_class)==1);        
    gt=ones(DB_size,1)*(-1);
    gt(temp)=1;
    
    [so,si]=sort(sim(i,:),'descend');
   
    tp=gt(si)>0;
    fp=gt(si)<0;
    
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/sum(gt>0);
    prec=tp./(fp+tp);

    % compute average precision
    ap=0;
    for t=0:0.1:1
        p=max(prec(rec>=t));
        if isempty(p)
            p=0;
        end
        ap=ap+p/11;
    end
    AP(i)=ap;     
end

mAP=zeros(size(AP,2),1);
for i=1:3:size(AP,2)-3+1
    mAP(i)=(AP(i)+AP(i+1)+AP(i+2))/3;
end

avg_score=sum(mAP)/num_class
