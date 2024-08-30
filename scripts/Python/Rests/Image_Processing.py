import os
from pyzbar.pyzbar import decode
from PIL import Image

def read_qr_code_and_rename_images(path_to_images):
    """Decode QR codes in images and rename the images based on the QR code data."""
    for filename in os.listdir(path_to_images):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(path_to_images, filename)
            img = Image.open(filepath)
            qr_codes = decode(img)
            img.close()
            if qr_codes:
                new_filepath = os.path.join(path_to_images, qr_codes[0].data.decode() + '.jpg')
                os.rename(filepath, new_filepath)

def rename_files_according_to_date(directory, date_suffix):
    """Rename files by appending a date suffix."""
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            num = filename.split('_')[0]
            new_filename = f"{num}_{date_suffix}.jpg"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

def rotate_image_based_on_exif(img_path):
    """Rotate images based on EXIF orientation data."""
    image = Image.open(img_path)
    try:
        exif_data = image._getexif()
        if exif_data and 274 in exif_data:  # 274 is the tag for Orientation
            orientation = exif_data[274]
            if orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif orientation == 8:
                image = image.transpose(Image.ROTATE_90)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def process_images_in_folder(folder_path):
    """Rotate images in a folder based on EXIF data."""
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = rotate_image_based_on_exif(img_path)
            img.save(img_path)

def crop_images_to_root_and_shoot(input_folder):
    """Crop images into root and shoot sections and save them in separate folders."""
    root_folder = os.path.join(input_folder, 'root')
    shoot_folder = os.path.join(input_folder, 'shoot')
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(shoot_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            
            # Root crop
            root_crop = image.crop((1550, 2780, 1550 + 3600, 2780 + 4800))
            root_name = filename.split('.')[0] + '_root.jpg'
            root_path = os.path.join(root_folder, root_name)
            root_crop.save(root_path)
            
            # Shoot crop
            shoot_crop = image.crop((304, 0, 304 + 5000, 0 + 3000))
            shoot_name = filename.split('.')[0] + '_shoot.jpg'
            shoot_path = os.path.join(shoot_folder, shoot_name)
            shoot_crop.save(shoot_path)

if __name__ == "__main__":
    # Set your image folder path
    folder_path = "C:/Users/shoai/Desktop/image/qr_code"
    
    # Step 1: Read QR codes and rename images
    read_qr_code_and_rename_images(folder_path)
    
    # Step 2: Rename files according to a specific date
    rename_files_according_to_date(folder_path, "11nov23")
    
    # Step 3: Rotate images based on EXIF data
    process_images_in_folder(folder_path)
    
    # Step 4: Crop images into root and shoot sections
    crop_images_to_root_and_shoot(folder_path)

    print("Image processing completed.")
