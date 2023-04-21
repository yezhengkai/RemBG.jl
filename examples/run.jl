using RemBG
using Images

exampledir(args...) = abspath(joinpath(dirname(pathof(RemBG)), "..", "examples", args...))

img_animal = load(exampledir("animal-1.jpg"))::Matrix{<:Colorant};
img_animal_num = copy(channelview(img_animal))::Array{<:Number};
img_cloth = load(exampledir("cloth-1.jpg"))::Matrix{<:Colorant};
img_cloth_num = copy(channelview(img))::Array{<:Number};

output_img_num = remove(img_animal_num);
output_img = remove(img_animal);  # If we do not pass an inference session to `remove`, it will use a new session containing the U2Net model.

# Use a specific model. The available models are U2Net, U2Netp, U2NetClothSeg, U2NetHumanSeg, Silueta, ISNetGeneralUse
session = new_session(U2Netp)
output_img_animal_num = remove(img_animal_num; session=session)
output_img_animal = remove(img_animal; session=session)

session = new_session(U2NetClothSeg)
output_img_cloth_num = remove(img_cloth_num; session=session)
output_img_cloth = remove(img_cloth; session=session)

session = new_session(U2NetHumanSeg)
output_img_animal_num = remove(img_animal_num; session=session)
output_img_animal = remove(img_animal; session=session)

session = new_session(Silueta)
output_img_animal_num = remove(img_animal_num; session=session)
output_img_animal = remove(img_animal; session=session)

session = new_session(ISNetGeneralUse)
output_img_animal_num = remove(img_animal_num; session=session)
output_img_animal = remove(img_animal; session=session)

# Return mask
output_img_animal = remove(img_animal; only_mask=true)  # return mask

# Apply background color
output_img_animal = remove(img_animal; bgcolor=(255, 0, 0, 255))
