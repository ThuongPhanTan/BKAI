#export PYTHONPATH= ${PWD}

#first run
#python3 setup.py install

#Predict
CUDA_VISIBLE_DEVICES="0" \
python3 demo/inference.py \
--config-file ./configs/BAText/VinText/attn_R_50.yaml \
--input ../../Data/image_private_process/ImageProcess \
--output ../../Results/abcnet \
--opts MODEL.WEIGHTS ./weights/weight_abcnet-dict-guided_final.pth

