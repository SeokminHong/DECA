for f in /mnt/d/ML/RAVDESS/Song/*.zip
  do
    echo "Processing $f"
    python demos/demo_normalize.py $f --dataset=ravdess --savefolder=/mnt/d/ML/output/ravdess/song/
  done

for f in /mnt/d/ML/RAVDESS/Speech/*.zip
  do
    echo "Processing $f"
    python demos/demo_normalize.py $f --dataset=ravdess --savefolder=/mnt/d/ML/output/ravdess/speech/
  done