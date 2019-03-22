function [H] = feature_quantize(label_num, numClusters,descr, centers)
%Assign the number of clusters
H=[];
for j=1:(4*label_num)
    h = zeros(1,numClusters);
    for i=1:size(descr{j},2)
    [~, k] = min(vl_alldist(single(descr{j}(:,i)), centers)) ;
  %  col_h=[col_h k];
    h(k) = h(k) + 1;  
    end
    h = h./sum(h);
    H=[H;h];
end
end

