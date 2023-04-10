
import multiprocessing
from fastcore.all import *
from fastai.vision.all import *


def train_model(path):
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    dls.show_batch(max_n=6)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(epochs=4)
    learn.save(Path("my_model_test"))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    path = Path('bass_or_not')
    train_model(path)

# epoch     train_loss  valid_loss  error_rate  time
# 0         1.019608    0.736622    0.292724    04:06
# epoch     train_loss  valid_loss  error_rate  time
# 0         0.676261    0.622453    0.228426    05:37
# 1         0.479550    0.530361    0.179357    05:21
# 2         0.292782    0.491161    0.167513    05:18

