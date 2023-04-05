import pydicom
import cv2
import os

# set the path to the folder containing the DICOM images
folder_path = "bad(.dcm)"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".dcm"):
        try:
            ds = pydicom.dcmread(os.path.join(folder_path, file_name))
        except pydicom.errors.InvalidDicomError:
            continue

        data = ds.pixel_array
        image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # loop over a range of angles in degrees
        for angle in range(-5, 5):
            # calculate the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1.0)

            # apply the rotation matrix to the image
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # adjust the brightness of the rotated image
            bright_image = cv2.convertScaleAbs(rotated, alpha=1.2, beta=0)

            # create a new folder for each PNG image
            folder_name = "result"
            os.makedirs(folder_name, exist_ok=True)

            # save the enhanced image as a PNG file inside the folder
            file_prefix = os.path.splitext(file_name)[0]
            cv2.imwrite(os.path.join(folder_name, f"{file_prefix}_{angle}.png"), bright_image)
