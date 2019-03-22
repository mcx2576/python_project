function [descr_test] = sift_all_test(label_num_test,binSize, magnif, imds_test,sift_function)
%Run sift for all test images

%Create list for TEST descriptors
d_a_test=zeros(label_num_test,1);
d_b_test=zeros(label_num_test,1);
d_c_test=zeros(label_num_test,1);
d_d_test=zeros(label_num_test,1);
descr_test = {};
for i=1:label_num_test
    %Read each image
    I_a_test=readimage(imds_test,i);
    I_b_test=readimage(imds_test,i+label_num_test);
    I_c_test=readimage(imds_test,i+2*label_num_test);
    I_d_test=readimage(imds_test,i+3*label_num_test);
    %Implement sift function
    d_a_test=sift(I_a_test, sift_function, binSize, magnif);
    d_b_test=sift(I_b_test, sift_function, binSize, magnif);
    d_c_test=sift(I_c_test, sift_function, binSize, magnif);
    d_d_test=sift(I_d_test, sift_function, binSize, magnif);
    descr_test{i}=d_a_test;
    descr_test{i+label_num_test}=d_b_test;
    descr_test{i+2*label_num_test}=d_c_test;
    descr_test{i+3*label_num_test}=d_d_test;
end
end

