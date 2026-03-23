clear all
close all
clc
time = 0;
fileFolder=fullfile(pwd,'SE0');
fileFolder1=fullfile(pwd,'SE4');

files=dir(fullfile(fileFolder,'*.dcm'));
files1=dir(fullfile(fileFolder1,'*.dcm'));

names={files.name};
[~, reindex] = sort( str2double(regexp({files.name}, '\d+', 'match', 'once' )));
names = names(reindex) ;
names1={files1.name};
[~, reindex1] = sort( str2double(regexp({files1.name}, '\d+', 'match', 'once' )));
names1 = names1(reindex1) ;
CT(:,:,1)=dicomread(fullfile(fileFolder, "IM0.dcm"));
for i=(length(names)):-1:1
    fname=fullfile(fileFolder,names{i});
    CT(:,:,i)=uint16(dicomread(fname));
end
CT(:,:,length(names)+2)=dicomread(fullfile(fileFolder, "IM245.dcm"));

PET(:,:,1)=dicomread(fullfile(fileFolder1, "IM0.dcm"));
for i=length(names1):-1:1
    fname1=fullfile(fileFolder1,names1{i});
    PET(:,:,i)=uint16(dicomread(fname1));
end
PET(:,:,length(names)+2)=dicomread(fullfile(fileFolder1, "IM245.dcm"));


infoCT = dicominfo(fullfile(fileFolder,names{1}));
infoPET=dicominfo(fullfile(fileFolder1,names1{1}));

tic
[optimizer,metric] = imregconfig('multimodal');

%Rfixed  = imref3d(size(CT),infoCT.Pixel,infoCT.PixelSpacing(1),infoCT.SliceThickness);
%Rmoving = imref3d(size(PET),infoPET.PixelSpacing(2),infoPET.PixelSpacing(1),infoPET.SliceThickness);
 
optimizer.InitialRadius = 0.0001;
optimizer.MaximumIterations = 1;
movingRegisteredVolume = imregister(PET, CT, 'similarity', optimizer, metric);
time=toc;
disp(toc)

for i=1:1:(length(names)+2)
    s0='Output/';
    s='IM';
    s1='.dcm';
    if (i == 1 || i == 248)
        disp(i)
    else
        disp(i)
        s2=num2str(i-1);
        dicomwrite(movingRegisteredVolume(:,:,i),strcat(s0,s,s2,s1));
    end
end
disp('time required is: ')
disp(time)
disp('sec')
beep
