# We took the following tutorial to implement the Dataloader for FiftyOneTorch and changed it so 
# it can be used for our model. The link for the tutorial is
# https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb#scrollTo=IRR2fVMlWeKI
import torch
import fiftyone.utils.coco as fouc
from PIL import Image
import numpy as np

class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the 
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """
    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("RGB")

        # Get all the bounding boxes.
        targets = sample.ground_truth.detections
        
        boxes = []
        labels = []
        
        for tar in targets:
            tar_id = self.labels_map_rev[tar.label]

            bbox = tar.bounding_box
            
            bbox[0] = bbox[0] + bbox[2]/2
            bbox[1] = bbox[1] + bbox[3]/2
            
            boxes.append(bbox)
            labels.append(tar_id)
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes
    
def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    
    data = torch.stack(data)

    tar_out = []
    for tar_idx, tar in enumerate(target):

        boxes = tar['boxes']
        clss = tar['labels']
        for box, cls in zip(boxes, clss):
            
            tar_out.append(torch.cat([torch.Tensor([tar_idx]), cls.view([1]), box]))
            
    tar_out = torch.stack(tar_out)
        
    return [data, tar_out]