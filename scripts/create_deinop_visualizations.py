import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random

# !!!!!!
# FOR ILLUSTRATION PURPOSES ONLY
# !!!!!!

def generate_bg_color():
    r = random.randint(50, 230)
    g = random.randint(50, 230)
    b = random.randint(50, 230)
    
    while max(r, g, b) - min(r, g, b) < 50:
        r = random.randint(50, 230)
        g = random.randint(50, 230)
        b = random.randint(50, 230)
    
    return (r, g, b)

def generate_fg_color():
    r = random.randint(0, 128)
    g = random.randint(0, 128)
    b = random.randint(0, 128)
    
    while max(r, g, b) - min(r, g, b) < 50:
        r = random.randint(0, 128)
        g = random.randint(0, 128)
        b = random.randint(0, 128)
    
    return (r, g, b)

def word_to_array(word):
    # Load the default font
    font = ImageFont.load_default()
    
    # List to hold each character's image array
    char_arrays = []
    
    for char in word:
        # Create a temporary image to determine the character's bounding box
        temp_img = Image.new('RGB', (10, 10), (255, 255, 255))
        draw = ImageDraw.Draw(temp_img)
        draw.text((1, -1), char, font=font, fill=(0, 0, 0))
        
        new_size = (256, 256)
        scaled = temp_img.resize(new_size, Image.NEAREST)
        char_img = Image.new('RGB', new_size, (255, 255, 255))
        paste_pos = (
            (256 - new_size[0]) // 2,
            (256 - new_size[1]) // 2
        )
        char_img.paste(scaled, paste_pos)

        char_array = np.array(char_img)

        thresh = 140
        char_array[char_array[:,:,0] < thresh, :] = 0
        char_array[char_array[:,:,0] >= thresh, :] = generate_bg_color()
        char_array[char_array[:,:,0] == 0, :] = generate_fg_color()
        
        char_arrays.append(char_array)
    
    return np.stack(char_arrays)

def plot(array, title= ""):
    # Get dimensions
    n_chars = array.shape[0]  # First dimension (number of characters)
    height = array.shape[1]   # Image height
    width = array.shape[2]    # Image width
    colors = array.shape[3]   # Number of color channels
    
    # Calculate number of rows needed (each row will show 3 color channels)
    colors_per_row = 3
    n_rows = (colors + colors_per_row - 1) // colors_per_row  # Ceiling division
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_chars, figsize=(n_chars * 2, n_rows * 2))

    fig.suptitle(f'{title}(Tensor Shape: {array.shape})')
    
    if n_rows == 1 or n_chars == 1:
        axes = np.atleast_2d(axes)
    
    # Pad the array if needed
    if colors % colors_per_row != 0:
        pad_size = colors_per_row - (colors % colors_per_row)
        padded_array = np.pad(array, 
                            ((0,0), (0,0), (0,0), (0,pad_size)),
                            mode='constant',
                            constant_values=0)
        padded_array[:, :, :, -pad_size:] = padded_array[:, :, :, -pad_size-1:-pad_size]
    else:
        padded_array = array
    
    for char_idx in range(n_chars):
        axes[0, char_idx].set_title(f'Batch {char_idx}')

    for row in range(n_rows):
        start_color = row * colors_per_row
        end_color = min((row + 1) * colors_per_row, colors)
        axes[row, 0].set_ylabel(f"Channel {start_color} to {end_color}", fontsize=10, labelpad=10)

    # Plot each character's color channels
    for char_idx in range(n_chars):
        for row in range(n_rows):
            start_color = row * colors_per_row
            end_color = (row + 1) * colors_per_row
            ax = axes[row, char_idx]
            color_img = padded_array[char_idx, :, :, start_color:end_color]
            ax.imshow(color_img.astype('uint8'))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def unfold(x, kernel_size, stride):
    N, C, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride
    
    # Calculate output dimensions
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1
    
    # Initialize output array
    output = np.zeros((N, C, kH, out_h, kW, out_w))
    
    # Fill the output array
    for i in range(kH):
        for j in range(kW):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * sH
                    w_start = w * sW
                    output[:, :, i, h, j, w] = x[:, :, 
                                               h_start + i, 
                                               w_start + j]
    output = np.permute_dims(output, (0,3,5,1,2,4))
    output = np.reshape(output, (N, out_h*out_w*C, kH, kW))
    
    return output

# Example usage:
if __name__ == "__main__":
    word = "SAFEDAG"
    array = word_to_array(word)
    
    plot(array, title="Original")

    # # permute
    permuted_dims = np.permute_dims(array,(0,2,1,3))
    plot(permuted_dims, title="permute dimensions 1 and 2")

    permuted_dims = np.permute_dims(array,(3,1,2,0))
    plot(permuted_dims, title="permute dimensions 0 and 3")

    # group/ungroup
    grouped_dims = np.reshape(array, (array.shape[0], array.shape[1], array.shape[2]*array.shape[3], 1))
    plot(grouped_dims, title="group dimensions 2 and 3")

    permute_arr = np.permute_dims(array,(0,1,3,2))
    grouped_dims = np.reshape(permute_arr, (permute_arr.shape[0], permute_arr.shape[1]*permute_arr.shape[2], permute_arr.shape[3], 1))
    plot(grouped_dims, title="permute then group dimensions 1 and 3")

    # slice
    sliced = array[1:-1, 10:170, 10:170, :]
    plot(sliced, title="slice")

    # pad
    padded = np.pad(array, pad_width=((0, 0), (30, 40), (20, 10), (0, 0)), mode='constant')  
    plot(padded, title="pad")

    # reduce
    reduced = np.mean(array, axis=3)
    reduced = np.reshape(reduced, (*reduced.shape, 1))
    plot(reduced, title="reduce mean color dimension")

    reduced = np.mean(array, axis=0)
    reduced = np.reshape(reduced, (1, *reduced.shape)) 
    plot(reduced, title="reduce average image")

    # select
    selected = array[0, :, :, :]
    selected = np.reshape(selected, (1, *selected.shape))
    plot(selected, title="select first batch")

    # repeat
    repeated = np.tile(array, reps=(1, 3, 4, 1))
    plot(repeated, title="unsqueze repeat group dimensions 1 and 2")

    # unfold/unfold
    unfolded = np.permute_dims(unfold(np.permute_dims(array,(0,3,1,2)), (160,160), (50,50)), (0,2,3,1))
    plot(unfolded, title="unfold (permuted and reshaped) dimensions 1 and 2")