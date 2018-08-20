#
# config file
#
dir="/data/vcea/matt.taylor/Projects/ras-object-detection/cyclegan"
program="CycleGAN.py"
modelFolder="models_horse2zebra"
logFolder="logs_horse2zebra"

# Emojis
#compressedDataset="emojis.zip"
#datasetFolder="emojis"
#A="emojis/Apple/*.png"
#B="emojis/Windows/*.png"
#Atest="emojis/Test_Apple/*.png"
#Btest="emojis/Test_Windows/*.png"
#width=72
#height=72
#channels=4
#patchsize=20
#batchsize=128
#gfd=16
#dfd=16
#residualblocks=3

compressedDataset="horse2zebra.zip"
datasetFolder="horse2zebra"
A="horse2zebra/trainA/*.jpg"
B="horse2zebra/trainB/*.jpg"
Atest="horse2zebra/testA/*.jpg"
Btest="horse2zebra/testB/*.jpg"
width=256
height=256
channels=3
patchsize=70
batchsize=15
gfd=32
dfd=64
residualblocks=6

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/18_Summer/Code/cyclegan/"
