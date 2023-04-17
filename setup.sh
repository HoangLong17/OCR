pip install opencv_python-4.5.5.64-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
pip install -r requirements.txt -i http://ocr:Mbocr123@10.1.12.177:8445/repository/pypi-public/simple  --trusted-host 10.1.12.177
if [ -d "/root/.cache/torch/hub/checkpoints" ] 
then
    echo "Directory /root/.cache/torch/hub/checkpoints exists." 
else    
    echo "Directory /root/.cache/torch/hub/checkpoints does not exists."
    mkdir /root/.cache/torch/hub/checkpoints
    echo "Create directory /root/.cache/torch/hub/checkpoints successfully."
fi
unzip vgg19_bn-c79401a0.zip
mv vgg19_bn-c79401a0.pth /root/.cache/torch/hub/checkpoints/vgg19_bn-c79401a0.pth