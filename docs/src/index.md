```@meta
CurrentModule = RemBG
```

# RemBG

This is a Julia implementation of the python package [rembg](https://github.com/danielgatis/rembg).

## Quick Start
```@example
using RemBG
using Images

exampledir(args...) = abspath(joinpath(dirname(pathof(RemBG)), "..", "examples", args...))

img = load(exampledir("animal-1.jpg"))
img_num = copy(channelview(img))

output_img_num = remove(img_num)
output_img = remove(img)  # default session use U2Netp model
save("test1.png", output_img)

# use specific session
session = new_session(U2Net)
output_img_num = remove(img_num; session=session)
output_img = remove(img; session=session)
save("test2.png", output_img)
```
**test1.png**
![](test1.png)
**test2.png**
![](test2.png)