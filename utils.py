# +
import os
from PIL import Image
import torch

from torchvision import transforms


# -

def removesuffix(content, suffix):
    if content.endswith(suffix):
        content = content[:-len(suffix)]
    return content

def findFiles(rootDir, suffix):
    files = []
    for r, d, f in os.walk(rootDir):
        for file in f:
            if suffix in file:
                files.append(removesuffix(str(file), suffix))
    return files

def substringBefore(string, char):
    return string[:string.index(char)]

def createIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resizeAllImages(src, target):
    for dataset in ["train", "val"]:
        inputRoot = src + "leftImg8bit/" + dataset + "/"
        targetRoot = src + "gtFine/" + dataset + "/"
        inputSuffix = '_leftImg8bit.png'
        targetSuffix = '_gtFine_labelIds.png'

        reducedRoot = target + dataset + "/"

        createIfNotExist(reducedRoot + "input/")
        createIfNotExist(reducedRoot + "target/")

        files = findFiles(inputRoot, inputSuffix)
        #targetImages = findFiles(inputRoot, targetSuffix)

        i = 0;

        targetSize = (512, 256)

        for file in files:
            cityName = substringBefore(file, "_")

            input = Image.open(inputRoot + cityName + "/" + file + inputSuffix)
            target = Image.open(targetRoot + cityName + "/" + file + targetSuffix)

            input = input.resize(targetSize)
            target = target.resize(targetSize, Image.NEAREST)

            input.save(reducedRoot + "input/" + str(i).zfill(4) +  ".png")
            target.save(reducedRoot + "target/" + str(i).zfill(4) +  ".png")

            i += 1


def saveModel(model, root, epoch):
    torch.save(model, root + "/r2u_epoch_" + str(epoch) + ".model")

from IPython.display import clear_output
def displayTensorAsImage( tensor ):
    display(transforms.ToPILImage()( tensor ))


def preview(inputs, outputs, targets, epoch):
    colors = cityscapeColors()
    
    outputs = outputs.cpu()
    targets = targets.cpu()

    values, indices = outputs.max(dim=1)
    targetValues, targetIndices = targets.max(dim=1)

    #set areas with classes that are not be trained on to "unknown" class
    excluded = targetValues < 0.1
    targetIndices[excluded] = 19
    indices[excluded] = 19

    colorImage = torch.stack([colors[indices, 0 ], colors[indices, 1], colors[indices, 2]], dim=1)
    targetImage = torch.stack([colors[targetIndices, 0 ], colors[targetIndices, 1], colors[targetIndices, 2]], dim=1)

    clear_output(wait=False)
    print(epoch)
    displayTensorAsImage( inputs[0] )
    displayTensorAsImage( targetImage[0] )
    displayTensorAsImage( colorImage[0] )


# +
_colors = torch.tensor([
    [128, 64,128],
    [244, 35,232],
    [ 70, 70, 70],
    [102,102,156],
    [190,153,153],
    [153,153,153],
    [250,170, 30],
    [220,220,  0],
    [107,142, 35],
    [152,251,152],
    [ 70,130,180],
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32],
    [0,0,0]
]) / 255

def cityscapeColors():
    return _colors
# -


