%% Getting Frames from a video 
% This is just to view the properties of the movie
% movObj = mmreader('35_1.avi', 'Tag', 'My reader object');
% get(movObj)
% 
% %Read and play back the movie file 32_1.avi:
% movObj = mmreader('35_1.avi');
% 
% nFrames = movObj.NumberOfFrames;
% vidHeight = movObj.Height;
% vidWidth = movObj.Width;
% 
% imageData=read(movObj,[10000 11500]);%THis was done...need to enter new-one
% dim=size(imageData);
% finish=dim(4);
% count=100;
% for i=1:10:finish
%     grayData=rgb2gray(imageData(:,:,:,i));
%     address=strcat('C:\Users\Nilou\Desktop\ECE281B\Project\DataBase\frame',int2str(count),'.png');
%     imwrite(grayData,address);
%     count=count+1;
% end

%% Annotating the data for the training set
% counter=214
% for i=1:78
% address1='C:\Users\Nilou\Desktop\ECE281B\Project'
% address=strcat(address1,'\DataBase\neg\frame',int2str(i),'.png')
% figure;imshow(address)
% I=imcrop
% figure;imshow(I)
% imwrite(I,strcat(address1,'\Database\training\neg\frame',int2str(counter),'.png'))  
% counter=counter+1;    
% end
% counter

%% Adding the left-right reflections of the images
% for i=1:300
% address1='C:\Users\Nilou\Desktop\ECE281B\Project'
% address=strcat(address1,'\DataBase\training\pos\frame',int2str(i),'.png')
% I=imread(address)
% address3=strcat(address1,'\DataBase\training\pos\frame',int2str(i+300),'.png')
% imwrite(fliplr(I),address3)
% end

%% Resizing the patches to 128*128 
for i=1:600
address1='C:\Users\Nilou\Desktop\ECE281B\Project'
address=strcat(address1,'\DataBase\training\pos\frame',int2str(i),'.png')
I=imread(address)
address3=strcat(address1,'\DataBase\training\pos2\frame',int2str(i),'.png')
I_resized = imresize(I, [64 128])
imwrite(I_resized,address3) 
end