"""Download KITTI 2D object detection data (images + training labels)."""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URLS = {
    "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
}


def download(url, dest):
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"already present and valid: {dest.name}, skipping download")
        return
    if dest.exists():
        print(f"removing incomplete {dest.name}, re-downloading")
        dest.unlink()
    print(f"downloading {dest.name} from {url} ...")

    def hook(block, block_size, total):
        if block % 5000 == 0:
            print(f"  {block * block_size / 1e6:.0f} MB / {total / 1e6:.0f} MB")

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"done: {dest.name}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = DATA / name
        download(url, dest)
    image_dir = DATA / "training" / "image_2"
    label_dir = DATA / "training" / "label_2"
    if len(list(image_dir.glob("*.png"))) != 7481:
        print("extracting data_object_image_2.zip ...")
        with zipfile.ZipFile(DATA / "data_object_image_2.zip") as z:
            z.extractall(DATA)
    else:
        print("images already extracted, skipping")
    if len(list(label_dir.glob("*.txt"))) != 7481:
        print("extracting data_object_label_2.zip ...")
        with zipfile.ZipFile(DATA / "data_object_label_2.zip") as z:
            z.extractall(DATA)
    else:
        print("labels already extracted, skipping")
    images = sorted((DATA / "training" / "image_2").glob("*.png"))
    labels = sorted((DATA / "training" / "label_2").glob("*.txt"))
    print(f"images: {len(images)}, labels: {len(labels)}")
    assert len(images) == 7481 and len(labels) == 7481, "unexpected counts — check the download"
    print("KITTI data ready in data/training/")


if __name__ == "__main__":
    main()
