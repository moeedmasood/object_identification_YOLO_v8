import argparse


def test(input_image_path=None, output_image_path=None):
    from PIL import Image
    from ultralytics import YOLO
    import tensorflow as tf
    import cv2

    if not input_image_path:
        image_path = "image_data\\default.jpg"
        path_list = image_path.split("\\")
        file_name = path_list[-1]
        new_image_path = "image_data\\default_cropped_output.jpg"
    else:
        image_path = input_image_path
        path_list = image_path.split("\\")
        file_name = path_list[-1]

    if not output_image_path:
        new_image_path = "image_data\\cropped_" + file_name
    else:
        new_image_path = output_image_path

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on input image
    results = model(image_path)  # results list

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save("results.jpg")  # save image

    # storing confidence tensor in a separate tensor variable
    confidence_tensor = results[0].boxes.conf

    # this will find the index of the maximum value within a tensor
    maxindex = tf.argmax(confidence_tensor)

    # this will separate the values of the box that is of a maximum confidence
    max_confidence_box = results[0].boxes[maxindex.numpy()]

    print(
        "The box with maximum confidence is a: "
        + results[0].names[int(results[0].boxes[0].cls.numpy())]
    )
    print("Confidence Level: ", end="")
    print(results[0].boxes.conf[maxindex.numpy()].numpy())

    orig_image = cv2.imread(image_path)
    numpy_xyxy = max_confidence_box.xyxy[0].numpy()
    x, y, x2, y2 = map(int, numpy_xyxy)
    cropped_image = orig_image[y:y2, x:x2]
    cv2.imwrite(new_image_path, cropped_image)

    print(
        "Refer to the new file that contain the cropped image (path is): "
        + new_image_path
    )
    print(
        "Please also refer to file results.jpg for the original image with predictions"
    )
    img2 = cv2.imread(new_image_path)

    if img2 is not None:
        print("Switch to other window to see the cropped image....")
        cv2.imshow("Cropped Image", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: unable to load the image")


def prediction(input_image_path=None):
    from PIL import Image
    from ultralytics import YOLO
    import tensorflow as tf
    import cv2
    from colorama import Fore, Style

    if not input_image_path:
        image_path = "image_data\\default.jpg"

    else:
        image_path = input_image_path

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on input image
    results = model(image_path)  # results list

    print("\n Here is the summary of the predictions:")

    i = 0
    for mycls in results[0].boxes.cls:
        object_name = results[0].names[int(mycls.numpy())]
        confidence_percentage = round(results[0].boxes.conf[i].numpy() * 100)

        formatted_object_name = (
            f"{Fore.RED}{Style.BRIGHT}{object_name}{Style.RESET_ALL}"
        )
        formatted_confidence = (
            f"{Fore.RED}{Style.BRIGHT}{confidence_percentage}%{Style.RESET_ALL}"
        )

        print(f"{i+1}. The object identified is: {formatted_object_name}", end="")
        print(f" and the confidence% of this object is: {formatted_confidence}")
        i += 1


def main(arg1, arg2):
    print("Argument 1:", arg1)
    print("Argument 2:", arg2)
    test(arg1, arg2)
    prediction(arg1)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A script with optional command-line arguments."
    )

    # Add optional arguments
    parser.add_argument("--arg1", help="Optional argument 1")
    parser.add_argument("--arg2", help="Optional argument 2")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.arg1, args.arg2)
