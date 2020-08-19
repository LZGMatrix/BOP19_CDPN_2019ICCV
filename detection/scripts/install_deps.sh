# some other dependencies
set -x
sudo apt-get install libjpeg-dev zlib1g-dev
sudo apt-get install libopenexr-dev
sudo apt-get install openexr
sudo apt-get install python3-dev
sudo apt-get install libglfw3-dev libglfw3
sudo apt-get install libassimp-dev

# conda install ipython
pip install cython
pip install plyfile

pip install pycocotools  # or install the nvidia version which is cpp-accelerated
# git clone https://github.com/NVIDIA/cocoapi.git cocoapi_nvidia
# cd cocoapi_nvidia/PythonAPI
# make
# python setup.py build develop

pip install cffi
pip install setproctitle
pip install fastfunc
pip install meshplex
pip install OpenEXR
pip install vispy
pip install tabulate
pip install pytest-runner
pip install pytest
pip install ipdb
pip install tqdm
pip install numba
pip install mmcv
pip install imagecorruptions
pip install pyassimp==4.1.3  # 4.1.4 will cause egl_renderer SegmentFault
pip install pypng
pip install albumentations
pip install transforms3d
pip install pyquaternion
pip install torchvision
pip install open3d
pip install fvcore
pip install tensorboardX

pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# install kaolin
# (our custom kaolin with DIB renderer for real camera projection and
# batch rendering for different objects and scene rendering)

# install detectron2
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2 && pip install -e .

# install pytorch3d
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d && pip install -e .

# install adet  # https://github.com/aim-uofa/adet.git
# git clone https://github.com/aim-uofa/adet.git
# cd adet
# python setup.py build develop
