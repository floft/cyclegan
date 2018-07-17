#
# config file
#
dir="/data/vcea/matt.taylor/Projects/ras-object-detection/cyclegan"
program="CycleGAN.py"
compressedDataset="dataset.zip"
datasetFolder="emojis"
modelFolder="models"
logFolder="logs"

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/18_Summer/Code/cyclegan/"
