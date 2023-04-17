using Tar, Inflate, SHA

build_dir = abspath(joinpath(@__DIR__, "build"))

for tarfile in readdir(build_dir; join=true)
    println("Checking: " * basename(tarfile))
    println("git-tree-sha1: ", Tar.tree_hash(IOBuffer(inflate_gzip(tarfile))))
    println("sha256: ", bytes2hex(open(sha256, tarfile)))
end
