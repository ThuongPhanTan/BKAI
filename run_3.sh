cd Recognize/vietocr && sh predict.sh

cd /scene_text_pipeline

cd Results && python3 merge.py 

cd /scene_text_pipeline/Results

zip prediction.zip *.txt 

rm *.txt