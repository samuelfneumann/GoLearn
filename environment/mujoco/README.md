# MuJoCo

Package `mujoco` implements functionality for using environments that use the [MuJoCo](http://www.mujoco.org/) physics simulator. All environments in this package have continuous action spaces and continuous state spaces.

The `XML` files that describe [MuJoCo](http://www.mujoco.org/) environments are found in the `assets` directory. The following `XML` files were taken from [OpenAI Gym](https://github.com/openai/gym/tree/master/gym/envs/mujoco/assets):

- ant.xml
- half_cheetah.xml
- hopper.xml
- humanoidstandup.xml
- humanoid.xml
- inverted_double_pendulum.xml
- inverted_pendulum.xml
- point.xml
- pusher.xml
- reacher.xml
- striker.xml
- swimmer.xml
- thrower.xml
- walker2d.xml

## How To Use
The `mujoco` packge uses `cgo` to interface with [MuJoCo](http://www.mujoco.org/). 

You must first have all the required files for [MuJoCo](http://www.mujoco.org/) on your system and
a valid [MuJoCo](http://www.mujoco.org/) license key (if using MuJoCo 2.0 ore before). 
The license key should be named `mjkey.txt` and should be placed in `~/.mujoco`. This package
assumes you will be using [MuJoCo](http://www.mujoco.org/) 2.0 or later. It has not been tested
for any versions of [MuJoCo](http://www.mujoco.org/) prior to 2.0, but may still work with
earlier versions.

This pacakge needs to know where the [MuJoCo](http://www.mujoco.org/) shared library
is and where the [MuJoCo](http://www.mujoco.org/) header files are. There are two ways
to do this:

First, you can set the paths to the
[MuJoCo](http://www.mujoco.org/) shared library and the [MuJoCo](http://www.mujoco.org/)
`C` header files using the `LDFLAGS` and `CFLAGS` `cgo` directives in all respective files
in this package. For a *usual* [MuJoCo](http://www.mujoco.org/) install with all files in
`/home/$USER/.mujoco`, these should be set as:

```
// #cgo CFLAGS:  -I/home/$USER/.mujoco/mujoco200_linux/include
// #cgo LDFLAGS: -L/home/$USER/.mujoco/mujoco200_linux/bin
```

where `mujoco200_linux` should be replaced with the respective *type* of
[MuJoCo](http://www.mujoco.org/) version on your system.

The second way is to add the following lines to you `.bashrc` file or
the `.zshrc` file:
```
export CGO_CFLAGS="-I/home/$USER/.mujoco/mujoco200_linux/include"
export CGO_LDFLAGS="-L/home/$USER/.mujoco/mujoco200_linux/bin"
```
assuming that your version of [MuJoCo](http://www.mujoco.org/) is
`mujoco200_linux`. If it is not, replace that portion of the code above
with your respective version of [MuJoCo](http://www.mujoco.org/).

Note that while this package is in active development, all development `cgo`
directives have been left in the files in this package. These should be removed
or altered to suit your system before you run code from this package. In
particular, the `-I` option for `CFLAGS` and the `-L` option for `LDFLAGS`
should be removed or altered, depending on if you used option 1 or 2 above to
set up the package.
