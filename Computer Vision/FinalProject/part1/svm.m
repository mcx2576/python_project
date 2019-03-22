function [model] = svm(H, type, label_num)
%%Running the svm classification
% Create the label vector matrix
    label_vector=[];
if strcmp(type,'airplanes')==1
     for i=1:label_num
        label=1;
        label_vector=[label_vector;label];
    end
    for i=(label_num+1):(2*label_num)
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=(2*label_num+1):(3*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(3*label_num+1):(4*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
elseif strcmp(type,'cars')==1
   for i=1:label_num
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(label_num+1):(2*label_num)
        label=1;
        label_vector=[label_vector;label];  
    end
    for i=(2*label_num+1):(3*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(3*label_num+1):(4*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
elseif strcmp(type,'faces')==1
    for i=1:label_num
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(label_num+1):(2*label_num)
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=(2*label_num+1):(3*label_num)
        label=1;
        label_vector=[label_vector;label];
    end
    for i=(3*label_num+1):(4*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
else 
    for i=1:label_num
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(label_num+1):(2*label_num)
        label=0;
        label_vector=[label_vector;label];  
    end
    for i=(2*label_num+1):(3*label_num)
        label=0;
        label_vector=[label_vector;label];
    end
    for i=(3*label_num+1):(4*label_num)
        label=1;
        label_vector=[label_vector;label];
    end
end
% train the model
best = train(label_vector, H, '-C -s 0');
model = train(label_vector, H, sprintf('-c %f -s 0', best(1)));
end

