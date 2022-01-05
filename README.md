# OrthomosaicSLAM
 
To carry out drone-based aerial surveying for generating orthomosaic maps on the fly, this project explores the image processing stack required to achieve the same using the most economical hardware and software footprint. The project explores corner and blob-based feature extraction techniques followed by brute force and KNN based feature matching methods which are later used to generate a homography matrix for stitching images that are passed through a cascaded image mixer to generate orthomosaic maps of a given dataset.

Explanation and documentation: https://textzip.github.io/posts/Orthomosaic-SLAM/

## Results
![Image1](/images/city_input.jpg)

## Usage
```bash
python3 main.py -i PATH_TO_IMAGES -o OUTPUT_PATH
```

## Paramters
Will be updated soon. 
