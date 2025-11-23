import os
def parse_image_name(filename):
    # remove extension
    name = os.path.splitext(filename)[0]  # ex: "tshirt_black"

    # split if underscore exists
    if "_" in name:
        item, color = name.split("_", 1)
    else:
        item = name
        color = None

    return item, color


def scan_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            item, color = parse_image_name(file)

            results.append([
                item,
                color
            ])

    return results