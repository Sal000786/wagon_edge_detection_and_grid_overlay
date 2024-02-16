import cv2
import imutils
import numpy as np

img_path = "F:\\Salman codes\\Open_CV_Course\\Edge_Detection\\final_wagon_image2.png"
img = cv2.imread(img_path)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower Threshold:", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper Threshold:", "Trackbars", 0, 255, nothing)

while True:
    lower = cv2.getTrackbarPos("Lower Threshold:", "Trackbars")
    upper = cv2.getTrackbarPos("Upper Threshold:", "Trackbars")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray,5)
    edges = cv2.Canny(blurred_image, lower, upper)
    cv2.imshow("Edges Image", edges)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw contours on
    img_with_contours = img.copy()

    # Draw contours on the copy
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)  # Green color for contours
    cv2.imshow("Contours", img_with_contours)  # Display the image with contours

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

cv2.imshow("img_with contours",img_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


#CODE FOR FINDING RECTANGLE IN CONTOURS

contours2 = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours2:
    approx = cv2.approxPolyDP(contour, 5, False)
    if len(approx) >= 4:
        location = approx
        break
print(location)
print(type(location))




# mask = np.zeros(gray.shape, np.uint8)
# cv2.imshow("mask",mask)
# cv2.waitKey(0)
# new_image = cv2.drawContours(mask, [location], 0,255, -1)
# cv2.imshow("masked image contour",new_image)
# cv2.waitKey(0)
# new_image = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow("masked image",new_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# Assuming location is not None and has the correct format
if location is not None and len(location) >= 4:
    # Convert location to a NumPy array of integers
    location = np.array(location, dtype=np.int32)

    # Create a mask for the region of interest
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [location], 255)

    # Bitwise AND the mask with the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Find the bounding box of the region of interest
    x, y, w, h = cv2.boundingRect(location)

    # Crop the image using the bounding box
    cropped_image = result[y:y+h, x:x+w]

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Location not valid.")


grid_size = 50
point_position = (0, 0)

# Function to draw the grid and point on the image
def draw_grid_and_point(img):
    # Draw the grid
    for x in range(0, img.shape[1], grid_size):
        cv2.line(img, (x, 0), (x, img.shape[0]), (255, 255, 255), 1)
    for y in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, y), (img.shape[1], y), (255, 255, 255), 1)

    # Draw the point
    cv2.circle(img, point_position, 5, (0, 0, 255), -1)

# Mouse callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global point_position

    # If left mouse button is clicked, update the point position
    if event == cv2.EVENT_LBUTTONDOWN:
        grid_x = (x // grid_size) * grid_size + grid_size // 2
        grid_y = (y // grid_size) * grid_size + grid_size // 2
        point_position = (grid_x, grid_y)
        print(f"Point moved to: {point_position}")

# Create an image (replace 'your_image.jpg' with your image path)
cv2.namedWindow('Image with Grid and Point')

# Set the mouse callback function
cv2.setMouseCallback('Image with Grid and Point', mouse_callback)

while True:
    # Clone the image to keep the original intact
    img_with_grid_and_point = cropped_image.copy()

    # Draw the grid and point on the image
    draw_grid_and_point(img_with_grid_and_point)
    cv2.imshow('Image with Grid and Point', img_with_grid_and_point)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
