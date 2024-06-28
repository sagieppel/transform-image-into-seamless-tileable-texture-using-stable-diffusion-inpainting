# use turn image into tileable seamless using stable diffusion


import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
def display_im(img,title=""):
    # Display tiled image before processing
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()



def create_seamless_texture(input_image_path, output_image_path, output_grid_image_path, prompt , strength, guidance_scale, impact,width, steps, grid_size=1024):
    # Load the model (Stable Diffusion 2)
    model_id = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Load and preprocess the input image
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    # Create a tiled version of the input image
    tiled_image = Image.new("RGB", (1024, 1024))
    for i in range(2):
        for j in range(2):
            tiled_image.paste(init_image, (i * 512, j * 512))
  #  display_im(tiled_image,"tile image before")

    # Create a mask for the boundaries
    mask = Image.new("L", (1024, 1024), 0)
    draw = ImageDraw.Draw(mask)

    # Draw vertical lines
    draw.line([(511, 0), (511, 1024)], fill=255, width=width)
    draw.line([(512, 0), (512, 1024)], fill=255, width=width)

    # Draw horizontal lines mark boundary areas that will be inpaint by stable diffusion
    draw.line([(0, 511), (1024, 511)], fill=255, width=width)
    draw.line([(0, 512), (1024, 512)], fill=255, width=width)
    mask_np = np.asarray(mask)
    merge_mask =  np.zeros_like(mask_np,dtype=np.float32)
    while(mask_np.sum()>0): # will use to merge the original image and the stable diffusion output image
      mask_np=cv2.erode(mask_np,np.ones([3,3],np.uint8))
      merge_mask+=mask_np.astype(np.float32)/255*impact
    merge_mask/=merge_mask.max()
    #merge_mask=merge_mask[:, :, None]
    tile_origin= np.asarray(tiled_image).astype(np.float32)
    # cv2.imshow("grad mask",(merge_mask*255).astype(np.uint8))
    # cv2.waitKey()
   # display_im(mask, "mask")



    # run stable diffusion Generate the seamless texture


    result = pipe(
        prompt=prompt,
        image=tiled_image,
        mask_image=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=grid_size,
        width=grid_size
    ).images[0]

    # merge the generated image with the original image
    result_tile_np = np.asarray(result)
    result_tile_np = cv2.resize(result_tile_np,(tile_origin.shape[1],tile_origin.shape[0]))
    final = (result_tile_np*merge_mask[:, :, None] + tile_origin*(1-merge_mask)[:, :, None]).astype(np.uint8)
    result = Image.fromarray(final.astype(np.uint8))

    result.save(output_grid_image_path)
    # Display tiled image after processing
   # display_im(result, "resutlt")

    print(f"result image size: {result.size}")
    # Crop the result to get the seamless texture
    seamless = result.crop((256, 256, 768, 768))




    # Rearrange the seamless texture to match the original layout
    rearranged = Image.new("RGB", (512, 512))
    rearranged.paste(seamless.crop((256, 256, 512, 512)), (0, 0))
    rearranged.paste(seamless.crop((0, 0, 256, 256)), (256, 256))
    rearranged.paste(seamless.crop((0, 256, 256, 512)), (256,0))
    rearranged.paste(seamless.crop((256,0, 512,256)), (0, 256))

    # Save the rearranged image
    rearranged.save(output_image_path)



    print(f"Seamless texture saved to {output_image_path}")

    # Verify the saved image
    saved_image = Image.open(output_image_path)
    print(f"Saved image size: {saved_image.size}")


if __name__ == "__main__":
    input_image_path = "in_image.jpg" # input image to be converted
    output_image_path = "converted_image.jpg" # output converted image
    output_grid_image_path = "converted_grid_image.jpg" # output converted image tiled as grid
    prompt = "seamless tileable texture hd"  # text prompt describing the blending area
    strength=0.9 # how much the image will change at boundaries
    guidance_scale=7.5 # how much the prompt will effect the image
    impact=0.6 # how much will the change effect the image (weighted average between input and output
    width=40 # width of boundary
    steps= 45 # number of diffusion step
    create_seamless_texture(input_image_path, output_image_path,output_grid_image_path,prompt,strength, guidance_scale,impact,width,steps)