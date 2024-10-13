# Create an image with only the cloudy pixels visible and other areas blacked out
filtered_cloud_image = np.zeros_like(new_image_with_clouds)

# Apply the cloud mask to highlight only the cloud pixels
filtered_cloud_image[cloud_mask_new_image] = new_image_with_clouds[cloud_mask_new_image]

# Convert back to an image
filtered_cloud_image_result = Image.fromarray(filtered_cloud_image)

# Save the filtered result to display
filtered_cloud_image_path = "/mnt/data/filtered_cloud_image.png"
filtered_cloud_image_result.save(filtered_cloud_image_path)

filtered_cloud_image_path