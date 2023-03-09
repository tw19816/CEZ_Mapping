import os
import Python.preprocessing as preprocessing
from Python.config import Config

def test_get_label_pixel_maps():
    label_to_rgb, rgb_to_label = preprocessing.get_label_pixel_maps(
        os.path.join(Config.root_path, "Random", "test_labelmap.txt")
    )
    for label in label_to_rgb.keys():
        rgb = label_to_rgb.get(label)
        assert label == rgb_to_label.get(rgb), \
            "Maps are not inverses."
    heath_output = label_to_rgb.get("heath") 
    assert heath_output == "240,120,240", \
        f"Heath should map to 240,120,240 but maps to {heath_output}"
 

def test_get_index_rgb_maps():
    rgb_to_index, index_to_rgb = preprocessing.get_rgb_index_maps(
        os.path.join(Config.root_path, "Random", "test_labelmap.txt"),
        os.path.join(Config.root_path, "Random", "test_index_labelmap.txt")
    )
    for rgb in rgb_to_index.keys():
        index = rgb_to_index.get(rgb)
        rgb_elemnts = index_to_rgb.get(index)
        assert rgb == ",".join(rgb_elemnts.astype(str)), "Maps are not inverses"
    zero_output = index_to_rgb.get(0)
    assert all(zero_output == (0, 0, 0)), \
        f"Index 0 should map to 0,0,0 but maps to {zero_output}"



if __name__ == "__main__":
    test_get_label_pixel_maps()
    test_get_index_rgb_maps()