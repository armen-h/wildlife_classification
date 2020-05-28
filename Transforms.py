import torch


class HorizontalFlip(object):

    def __call__(self, sample):
        image = sample['image']
        image_class = sample['class']
        image_name = sample['image_name']

        image = torch.flip(image, 1)
        return {'image': image, 'class': image_class, 'image_name': 'flipped_' + image_name}
