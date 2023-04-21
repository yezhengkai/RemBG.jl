```@meta
CurrentModule = RemBG
```

# RemBG

This is a Julia implementation of the python package [rembg](https://github.com/danielgatis/rembg).

## Quick Start
```@example quick_start
using RemBG
using Images

exampledir(args...) = abspath(joinpath(dirname(pathof(RemBG)), "..", "examples", args...))

img = load(exampledir("animal-1.jpg"))
img_num = copy(channelview(img))

output_img_num = remove(img_num)
output_img = remove(img)  # If we do not pass an inference session to `remove`, it will use a new session containing the U2Net model.
```

```@example quick_start
# Use a specific model. The available models are U2Net, U2Netp, U2NetClothSeg, U2NetHumanSeg, Silueta, ISNetGeneralUse
session = new_session(U2Netp)
output_img_num = remove(img_num; session=session)
output_img = remove(img; session=session)
```

```@example quick_start
output_img = remove(img; only_mask=true)  # return mask
```

```@example quick_start
output_img = remove(img; bgcolor=(255, 0, 0, 255))  # apply background color
```