package = "culab"
version = "scm-1"

source = {
   url = "git://github.com/fartashf/culab.torch",
   tag = "master"
}

description = {
   summary = "A Cuda implementation of image.rgb2lab",
   detailed = [[
   	    A Cuda implementation of image.rgb2lab
   ]],
   homepage = "https://github.com/fartashf/culab.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "cutorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN)
]],
   install_command = "cd build && $(MAKE) install"
}
