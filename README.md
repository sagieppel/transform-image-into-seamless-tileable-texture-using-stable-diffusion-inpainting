# transform-image-into-seamless-tileable-texture-using-stable-diffusion-inpainting
Change the texture image to seamless,  such that if can be tiled seamlessly with itself with no stitching lines. Meaning the opposite side of the image merges seamlessly with each other. This is done by changing the boundary of the image using stable diffusion inpainting option on the boundaries.

# convert-image-into-seamless-tileable-texture
Turn a texture image, such that it can be tiled seamlessly into uniform texture. Basically, change the image boundary regions to hide the stitching line in the image's native patterns.
This is taking the boundaries of the image and changing them so that opposite sides of the image will merge seamlessly, this is done by using the image topography to create stitching lines that follow the image's native patterns and as such much less conscious and harder to detect.


## original image. 
![](sa_266035_15_Score_5229_TileSize39_Texture.jpg)


##  converted image.
![](sa_266472_5_Score_5119_TileSize39_Texture.jpg)
 

# setting
install stable diffusion:
[https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)

 
# run on image
in:  make_image_seamless.py


set parameters:

input_image_path =  path input image to be converted

output_image_path =  path to output converted image

output_grid_image_path =  path output converted image tiled as grid
 
# more parameters

 prompt =   text prompt describing the blending area
 
 strength =  how much the image will change at boundaries
 
 guidance_scale =  how much the prompt will affect the image
 
 impact =  how much will the change affect the image (weighted average between input and output
 
 width =  width in pixels of boundary region that will be changed
 
 steps =   number of diffusion step


# Alternative resources for making texture images seamless
1) [https://github.com/sagieppel/convert-image-into-seamless-tileable-texture](https://github.com/sagieppel/convert-image-into-seamless-tileable-texture)
2) [https://github.com/rtmigo/img2texture](https://github.com/rtmigo/img2texture) 

# license 
Code is under cc0 but images in in_dir and display taken from the segment anything repository.
