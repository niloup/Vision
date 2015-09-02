%%%%compute the normalized size of the regions and the normalized position of the regions

% matlabpool
% clear all
% clc;

imageFolder='PATH';
fileList = dir(fullfile(imageFolder, '*.gif'))';

DB_size=4952;

parfor i=1:DB_size
    i
    image=imread([imageFolder fileList(1,i).name]); 
    [R,C]=size(image);
    
    num_nodes=size(unique(image),1);
    
    temp=unique(image);
    for count=1:num_nodes
       image(image==temp(count))=count-1; 
    end
    
    temp_size=[];
    for j=1:num_nodes
       index2=find(image==j-1);
       temp_size(j)=size(index2,1)/(R*C);
    end
    Region_size{i}=temp_size;
    
    
    temp_position=[];
    for j=1:num_nodes
       [x y]=find(image==j-1);
       temp_position(j,:)=[sum(x)/(size(x,1)*R) sum(y)/(size(y,1)*C)];
    end
    Region_position{i}=temp_position;
       
end


save('Region_size_GM','Region_size','-v7.3');
save('Region_position_GM','Region_position','-v7.3');

