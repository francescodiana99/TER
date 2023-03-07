#%%
from pdf2image import convert_from_path
import numpy as np
import pytesseract
import cv2

#%%

def preprocess(image):
    """Preprocess the image, applying Otsu thresholding, opening by reconstruction and median filtering"""

    # print(image.shape)
    # (h,w) = image.shape[:2]
    # aspect_ratio = h/w

    # defining window size only to display purposes, I know it is bad, but i will not save the image with these dimensions
    # window_width = 600
    # window_height = int(window_width * aspect_ratio)


    # conversion of the image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # thresholding with Otsu's method
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # application of morphological operations, in particular opening by reconstruction and median filtering
    # in this way it is possible to remove noise from the images, trying to reconstruct  the original shape of letters

    def imreconstruct(marker: np.ndarray, mask: np.ndarray, kernel_dim=10):
        """Implementation of reconstruction by dilation morphological operation.
        The marker is iteratively expanded until it reaches stability
        params:
        marker: Image containing the initial seed, starting point for the reconstruction
        mask: Image that constrains the reconstruction"""

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dim,kernel_dim))

        while True:
            rec_by_dil = cv2.morphologyEx(marker, cv2.MORPH_DILATE, kernel)
            cv2.bitwise_and(src1=rec_by_dil, src2=mask, dst=rec_by_dil)

            if(marker == rec_by_dil).all():
                return rec_by_dil
            marker = rec_by_dil

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    marker = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    op_by_rec = imreconstruct(marker=marker, mask=thresh)

    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # removal of salt and pepper noise
    med_filtered = cv2.medianBlur(op_by_rec, 3)
    processed = cv2.bitwise_not(med_filtered)

    return processed

# convert the pdf in a series of jpeg files
pdf = r"./data/test.pdf"
pages = convert_from_path(pdf,dpi=350)
# create a text file where I can save text extracted frim the images
test_file = open('./data/test.txt', 'w')
page_number = 1

for page in pages:
    image_name = "Page" + str(page_number) + ".jpg"
    # save each page of the pdf in a jpeg file
    page.save("./data/test_images/" + image_name , format="JPEG")
    image = cv2.imread("./data/test_images/Page" + str(page_number) + ".jpg")
    # preprocess the image
    preprocess(image=image)
    # save the processed image
    cv2.imwrite("./data/test_images/Page" + str(page_number) + ".jpg", img=image)
    #text exrtaction using tesseract
    # parameter psm indicates how text is extracted, I could have put flag -l fra, but I had worse results
    text = str(pytesseract.image_to_string(image, config='--psm 1 '))
    # save text in the test,txt file
    test_file.write(text)
    test_file.write('\n')
    test_file.write('\n')
    test_file.write('\n')
    test_file.write("------------------------    "+ str(page_number) + "    ------------------------")
    test_file.write('\n')
    test_file.write('\n')
    test_file.write('\n')


    page_number = page_number + 1
test_file.close


#%%
image = cv2.imread("./data/test_images/Page1.jpg" )
text = str(pytesseract.image_to_string(image, config='--psm 1 '))

print(text)