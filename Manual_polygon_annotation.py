import cv2
import numpy as np

points = []
image_path = "D:\\task\\archive\\PennFudanPed\\PNGImages\\FudanPed00001.png"  
image = cv2.imread(image_path)
clone = image.copy()

def draw_polygon(event, x, y, flags, param):
    global points, image
    
    if event == cv2.EVENT_LBUTTONDOWN:  
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Polygon Annotation", image)

    elif event == cv2.EVENT_RBUTTONDOWN:  
        if points:
            points.pop()
            image = clone.copy()
            for i, p in enumerate(points):
                cv2.circle(image, p, 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(image, points[i - 1], points[i], (0, 255, 0), 2)
            cv2.imshow("Polygon Annotation", image)

def main():
    global image
    cv2.namedWindow("Polygon Annotation")
    cv2.setMouseCallback("Polygon Annotation", draw_polygon)

    while True:
        cv2.imshow("Polygon Annotation", image)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Press Enter to close the polygon
            if len(points) > 2:
                cv2.line(image, points[-1], points[0], (0, 255, 0), 2)  # Close polygon
                cv2.imshow("Polygon Annotation", image)
            print("Annotated Points:", points)
        
        elif key == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
