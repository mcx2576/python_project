%run('C:\Users\mcx25\Downloads\MathWorks\R2017b\archives\common\matlab\vlfeat-0.9.21\toolbox/vl_setup')
tempdir='';
outputFolder = fullfile(tempdir, 'Caltech4');
rootFolder = fullfile(outputFolder, 'ImageData');
categories = {'airplanes_train', 'cars_train', 'faces_train', 'motorbikes_train'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%Create training image list
label_num=100;
imds = splitEachLabel(imds, label_num, 'randomize');
airplanes = find(imds.Labels == 'airplanes_train', label_num);
cars=find(imds.Labels == 'cars_train', label_num);
faces=find(imds.Labels == 'faces_train', label_num);
motorbikes=find(imds.Labels == 'motorbikes_train', label_num);
%Create test image list
label_num_test=50;
categories_test = {'airplanes_test', 'cars_test', 'faces_test', 'motorbikes_test'};
imds_test = imageDatastore(fullfile(rootFolder, categories_test), 'LabelSource', 'foldernames');


%Choose the type of sift function and speciy parameters
%Users could choose between "gray", "RGBSIFT", "rgbSIFT", "KeyPointSIFT"
%and "opponentSIFT"
sift_function="rgbSIFT";
binSize = 8 ;
magnif = 3 ;

% Run sift to find descriptors for training images
[descr] = sift_all(label_num,binSize, magnif, imds,airplanes, cars, faces, motorbikes, sift_function);

% Find each category of 50 images for testing images
[descr_test] = sift_all_test(label_num_test,binSize, magnif, imds_test,sift_function);
%%Runing the k-means to build up the Bow
num_bow=400;
des_bow={};
for k=1:(num_bow/4)
    des_bow{k}=descr{k};
    des_bow{k+num_bow/4}=descr{k+label_num};
    des_bow{k+num_bow/2}=descr{k+label_num*2};
    des_bow{k+num_bow*3/4}=descr{k+label_num*3};
end
%Convert the cell into matrix
M=single(cell2mat(des_bow));
numClusters=800;
num_point = 400;
[centers,assignments] = vl_kmeans(M, numClusters); 

%Create the frequency matrix for training set
[H] = feature_quantize(label_num, numClusters,descr, centers);

%Create the frequency matrix for test set

[H_test] = feature_quantize(label_num_test, numClusters,descr_test, centers);


%%Running the svm classification
% train the model and test the results
H=sparse(H);
[model_airplanes, model_cars, model_faces, model_motorbikes] = svm_all_model(H,label_num);
%%Evaluate each model
H_test=sparse(H_test);
[predict_label_a,accuracy_a, dec_values_a,predict_label_b,accuracy_b, dec_values_b, predict_label_c,accuracy_c, dec_values_c, predict_label_d,accuracy_d, dec_values_d, label_vector_a, label_vector_b, label_vector_c, label_vector_d] = predict_all_model(model_airplanes, model_cars, model_faces, model_motorbikes,H_test);

%Calculate the average precision
[AP_a, index_ranked_a] = calc_AP(label_vector_a, dec_values_a, "airplanes");
[AP_b, index_ranked_b] = calc_AP(label_vector_b, dec_values_b, "cars");
[AP_c, index_ranked_c] = calc_AP(label_vector_c, dec_values_c, "faces");
[AP_d, index_ranked_d] = calc_AP(label_vector_d, dec_values_d, "motorbikes");
print_lines(imds_test, index_ranked_a, index_ranked_b, index_ranked_c, index_ranked_d);
%