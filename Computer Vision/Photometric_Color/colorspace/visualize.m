function visualize(input_image)

%Extract values from each channel
channel_1 = input_image(:,:,1);
channel_2 = input_image(:,:,2);
channel_3 = input_image(:,:,3);
%Plot each channel as well as the original picture in the same figure
a = size(input_image)
if a(3)==4
    channel_4=input_image(:,:,4);
else
    channel_4=input_image;
end
subplot(2,2,1);
imshow(channel_1);
subplot(2,2,2);
imshow(channel_2);
subplot(2,2,3);
imshow(channel_3);
subplot(2,2,4);
imshow(channel_4);

end

