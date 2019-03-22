function [] = predict(model,type, H_test)
% Create the label vector matrix
if strcmp(type,'airplanes')==1
    label_vector=[];
    for i=1:50
        label=1;
        label_vector=[label_vector;label];
    end
    for i=51:100
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=101:150
        label=0;
        label_vector=[label_vector;label];
    end
    for i=151:200
        label=0;
        label_vector=[label_vector;label];
    end
elseif strcmp(type,'cars')==1
     label_vector=[];
    for i=1:50
        label=0;
        label_vector=[label_vector;label];
    end
    for i=51:100
        label=1;
        label_vector=[label_vector;label];  
    end
    for i=101:150
        label=0;
        label_vector=[label_vector;label];
    end
    for i=151:200
        label=0;
        label_vector=[label_vector;label];
    end
elseif strcmp(type,'faces')==1
     label_vector=[];
    for i=1:50
        label=0;
        label_vector=[label_vector;label];
    end
    for i=51:100
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=101:150
        label=1;
        label_vector=[label_vector;label];
    end
    for i=151:200
        label=0;
        label_vector=[label_vector;label];
    end
else 
     label_vector=[];
    for i=1:50
        label=0;
        label_vector=[label_vector;label];
    end
    for i=51:100
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=101:150
        label=0;
        label_vector=[label_vector;label];
    end
    for i=151:200
        label=1;
        label_vector=[label_vector;label];
    end
end

%[predict_label, accuracy, dec_values] = predict(label_vector, H_test, model);

end

