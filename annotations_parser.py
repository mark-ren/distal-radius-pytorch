import sys
import os
import json
from pycocotools.coco import COCO


RAW_ANNOT_PATH = 'datasets/annotations/distalradius.json'
PROCESSED_ANNOT_PATH = 'datasets/annotations/distalradius_processed.json'


def convert_coco_annots(annotations_file=RAW_ANNOT_PATH):
    coco_annotation = COCO(annotations_file)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    # print(f"Number of Unique Categories: {len(cat_ids)}")
    # print("Category IDs:")
    # print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    # print("Categories Names:")
    # print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    # query_supercategory = query_annotation["supercategory"]
    # print(
    #     f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    # )

    # Category Name -> Category ID.
    query_name = cat_names[0]
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    # print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds(catIds=[query_id])
    # print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    # Pick one image.
    # img_id = img_ids[2]
    # img_info = coco_annotation.loadImgs([img_id])[0]
    # img_file_name = img_info["file_name"]
    # img_path = img_info["path"]
    # print(
    #     f"Image ID: {img_id}, File Name: {img_file_name}, Image Path: {img_path}"
    # )

    # Get all the annotations for the specified image.
    # ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    # anns = coco_annotation.loadAnns(ann_ids)
    # print(f"Annotations for Image ID {img_id}:")
    # print(anns)

    # Use URL to load image.
    # im = Image.open(img_path.strip('/'))

    # Save image and its labeled version.
    # plt.axis("off")
    # plt.imshow(np.asarray(im))
    # plt.savefig(f"{img_id}.jpg", bbox_inches="tight", pad_inches=0)
    # Plot segmentation and bounding box.
    # coco_annotation.showAnns(anns, draw_bbox=True)
    # plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)
    # plt.show()

    # for id in img_ids:
    #     print(coco_annotation.loadImgs([id])[0]['file_name'])
    converted_data = []
    for i in img_ids:
        img_data = coco_annotation.loadImgs([i])[0]
        ann_ids = coco_annotation.getAnnIds(imgIds=[i], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        converted_datum = {}
        converted_datum['image_id'] = img_data['id']
        converted_datum['file_name'] = img_data['file_name']
        converted_datum['path'] = img_data['path']
        converted_datum['width'] = img_data['width']
        converted_datum['height'] = img_data['height']
        converted_datum['bboxes'] = []
        converted_datum['keypoints'] = []
        for a in anns:
            if a['bbox'] != [0, 0, 0, 0]:  # exception for bboxes and annotations not associated with each other
                convertedbbox = [a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                converted_datum['bboxes'].append(convertedbbox)
            converted_keypoints = []
            if 'keypoints' in a:
                for j in range(0, len(a['keypoints']), 3):
                    converted_keypoints.append([a['keypoints'][j], a['keypoints'][j + 1], 1])
                converted_datum['keypoints'].append(converted_keypoints)
        converted_data.append(converted_datum)
    return(converted_data)


def main(annotations_file=RAW_ANNOT_PATH):
    # change current working directory to this file's folder
    os.chdir(sys.path[0])
    processed_annots = convert_coco_annots(annotations_file)
    # with open(PROCESSED_ANNOT_PATH, "w") as outfile:
    # json.dump(processed_annots, outfile)
    for a in processed_annots:
        print(a)


main()
