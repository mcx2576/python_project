function print_lines(imds_test, index_ranked_a, index_ranked_b, index_ranked_c, index_ranked_d)
for k=1:200
    full_path = '<tr>';
%each rank for air plane
    path_a=imds_test.Files(index_ranked_a(k));
    path_a=split(path_a{1},"\");
    part_path_a=fullfile(path_a{end-3},'/',path_a{end-2},'/',path_a{end-1},'/',path_a{end});
%each rank for cars
    path_b=imds_test.Files(index_ranked_b(k));
    path_b=split(path_b{1},"\");
    part_path_b=fullfile(path_b{end-3},'/',path_b{end-2},'/',path_b{end-1},'/',path_b{end});
%each rank for faces
    path_c=imds_test.Files(index_ranked_c(k));
    path_c=split(path_c{1},"\");
    part_path_c=fullfile(path_c{end-3},'/',path_c{end-2},'/',path_c{end-1},'/',path_c{end});
%each rank for motorbike
    path_d=imds_test.Files(index_ranked_d(k));
    path_d=split(path_d{1},"\");
    part_path_d=fullfile(path_d{end-3},'/',path_d{end-2},'/',path_d{end-1},'/',path_d{end});
    line_part_a = strcat('<td><img src="', part_path_a, '" /></td>');
    line_part_b = strcat('<td><img src="', part_path_b, '" /></td>');
    line_part_c = strcat('<td><img src="', part_path_c, '" /></td>');
    line_part_d = strcat('<td><img src="', part_path_d, '" /></td>'); 
   full_path = strcat(full_path,line_part_a,line_part_b,line_part_c,line_part_d);
   full_path = strcat(full_path,'</tr>');
disp(full_path)
end
disp(full_path)
%for k = 1:200

 %  for i = 1:4
 %       pos_offset = ???; %depends on machine where it is run
 %       if i == 1
 %           path = paths_test_imgs(index_airplanes)(pos_offset:end);
 %       elseif i == 2
 %           path = paths_test_imgs(index_cars)(pos_offset:end);
 %       elseif i == 3
  %          path = paths_test_imgs(index_faces)(pos_offset:end);
   %     elseif i == 4
    %        path = paths_test_imgs(index_motorbikes)(pos_offset:end);
     %   end

      %  line_part_i = strcat('<td><img src="', path, '" /></td>');
       % line = strcat(line,line_part_i);
   % end    

  %  line = strcat(line,'</tr>');
   % disp(line);

end
