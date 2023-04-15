using RemBG
using Images

exampledir(args...) = joinpath(@__DIR__, args...);
img = load(exampledir("animal-1.jpg"))::Matrix{<:Colorant};
img_num = copy(channelview(img))::Array{<:Number};

output_img_num = remove(img_num);
output_img = remove(img);

session = new_session(U2Net);
output_img_num = remove(img_num; session=session);
output_img = remove(img; session=session);

session = new_session(U2NetHumanSeg);
output_img_num = remove(img_num; session=session);
output_img = remove(img; session=session);

session = new_session(ISNetGeneralUse);
output_img_num = remove(img_num; session=session);
output_img = remove(img; session=session);

# cloth
img = load(exampledir("cloth-1.jpg"))::Matrix{<:Colorant};
img_num = copy(channelview(img))::Array{<:Number};
session = new_session(U2NetClothSeg);
output_img_num = remove(img_num; session=session);
output_img = remove(img; session=session);
