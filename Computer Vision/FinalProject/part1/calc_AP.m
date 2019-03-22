function [AP, I] = calc_AP(labels, dec_values, choice)

% First create the ranking (from big to small)
if choice == "airplanes"
    [~,I] = sort(dec_values,'descend');
else 
    [~,I] = sort(dec_values,'ascend');
end
labels_ranked = labels(I);

num_imgs = 200;
m_c = 50;

AP = 0;
count = 0;
for k = 1:num_imgs
    % label is set to 1 for positive match, else 0
    if labels_ranked(k) == 1
        count = count + 1;
        AP = AP + double(count/k);
    end
end
AP = AP / m_c;

end




