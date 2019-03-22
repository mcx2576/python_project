function [descr] = sift_all(label_num,binSize, magnif, imds,airplanes, cars, faces, motorbikes, sift_function)
sift_function=sift_function;
binSize = binSize ;
magnif = magnif ;
d_a=zeros(label_num,1);
d_b=zeros(label_num,1);
d_c=zeros(label_num,1);
d_d=zeros(label_num,1);
descr = {};
for i=1:label_num
    %Read each image
    I_a=readimage(imds,airplanes(i));
    I_b=readimage(imds,cars(i));
    I_c=readimage(imds,faces(i));
    I_d=readimage(imds,motorbikes(i));
    %Implement sift function
    d_a=sift(I_a, sift_function, binSize, magnif);
    d_b=sift(I_b, sift_function, binSize, magnif);
    d_c=sift(I_c, sift_function, binSize, magnif);
    d_d=sift(I_d, sift_function, binSize, magnif);
    descr{i}=d_a;
    descr{i+label_num}=d_b;
    descr{i+2*label_num}=d_c;
    descr{i+3*label_num}=d_d;
end
end