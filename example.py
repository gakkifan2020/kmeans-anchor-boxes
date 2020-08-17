import glob
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import numpy as np

from kmeans import kmeans, avg_iou

data_type = 'coco'
ANNOTATIONS_PATH = "/home/zhangwei/PycharmProjects/DL/YoloV3_tf2/data/coco/annotations/instances_train2017.json"
CLUSTERS = 9

def id2name(coco):
    classes = dict()
    classes_id = []
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']

    for key in classes.keys():
        classes_id.append(key)
    return classes, classes_id

def load_dataset(path, types='voc'):
	dataset = []
	if types == 'voc':
		for xml_file in glob.glob("{}/*xml".format(path)):
			tree = ET.parse(xml_file)

			height = int(tree.findtext("./size/height"))
			width = int(tree.findtext("./size/width"))

			for obj in tree.iter("object"):
				xmin = int(obj.findtext("bndbox/xmin")) / width
				ymin = int(obj.findtext("bndbox/ymin")) / height
				xmax = int(obj.findtext("bndbox/xmax")) / width
				ymax = int(obj.findtext("bndbox/ymax")) / height

				dataset.append([xmax - xmin, ymax - ymin])

		return np.array(dataset)
	if types == 'coco':
		coco = COCO(path)
		classes, classes_id = id2name(coco)
		print(classes)
		print('class_ids:', classes_id)

		img_ids = coco.getImgIds()
		print(len(img_ids))

		for imgId in img_ids:
			i = 0
			img = coco.loadImgs(imgId)[i]
			height = img['height']
			width = img['width']
			i = i + 1
			if imgId % 500 == 0:
				print('process {} images'.format(imgId))
			annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
			anns = coco.loadAnns(annIds)
			# time.sleep(0.2)
			for ann in anns:
				if 'bbox' in ann:
					bbox = ann['bbox']
					'''
                     coco:
                    annotation: [x, y, width, height] 
                    '''
					ann_width = bbox[2]
					ann_height = bbox[3]

					# 偏移量
					ann_width = np.float64(ann_width / width)
					ann_height = np.float64(ann_height / height)
					dataset.append([ann_width, ann_height])
				else:
					raise ValueError("coco no bbox -- wrong!!!")

	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH, data_type)
data_nonzero=[]
for i in data:
    if i[0]!=0 and i[1]!=0:
        data_nonzero.append(i)
data_nonzero=np.array(data_nonzero)
out = kmeans(data_nonzero, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data_nonzero, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))