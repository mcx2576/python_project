function [predict_label_a,accuracy_a, dec_values_a,predict_label_b,accuracy_b, dec_values_b, predict_label_c,accuracy_c, dec_values_c, predict_label_d,accuracy_d, dec_values_d, label_vector_a, label_vector_b, label_vector_c, label_vector_d] = predict_all_model(model_airplanes, model_cars, model_faces, model_motorbikes,H_test)
%Run prediction for all models
type='airplanes';
[predict_label_a,accuracy_a, dec_values_a, label_vector_a] = prediction(model_airplanes,type, H_test);
type='cars';
[predict_label_b,accuracy_b, dec_values_b, label_vector_b] = prediction(model_cars,type, H_test);
type='faces';
[predict_label_c,accuracy_c, dec_values_c, label_vector_c] = prediction(model_faces,type, H_test);
type='motorbikes';
[predict_label_d,accuracy_d, dec_values_d, label_vector_d] = prediction(model_motorbikes,type, H_test);
end