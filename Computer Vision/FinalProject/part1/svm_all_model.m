function [model_airplanes, model_cars, model_faces, model_motorbikes] = svm_all_model(H,label_num)
%Generate svm model for all types
type='airplanes';
[model_airplanes] = svm(H, type, label_num);
type='cars';
[model_cars] = svm(H, type, label_num);
type='faces';
[model_faces] = svm(H, type, label_num);
type='motorbikes';
[model_motorbikes] = svm(H, type, label_num);
end

