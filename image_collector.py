
import multiprocessing
from duckduckgo_search import ddg_images
from time import sleep
from fastai.vision.utils import resize_images,download_images,verify_images
from fastai.data.transforms import get_image_files
from fastcore.all import *


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

def get_images(num):
    searches = 'bass guitar', 'electric guitar'
    path = Path('test')
    for o in searches:
        dest = (path / o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o}',max_images=num))
        sleep(10)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} on stage',max_images=num))
        sleep(10)

        resize_images(path / o, max_size=400, dest=path / o)
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(len(failed))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    get_images(10)
