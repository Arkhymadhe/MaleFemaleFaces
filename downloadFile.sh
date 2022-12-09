#!/bin/bash
# Download and unzip file
#wget https://www.dropbox.com/s/rbajpdlh7efkdo1/male_female_face_images.zip
#unzip male_female_face_images.zip

# Rename unzipped folder
#mkdir dataset
#mv male_female_face_images dataset

# Move trained artefacts to appropriate folder
mv scripts/saved_generator.ckpt artefacts/saved_generator.ckpt
mv scripts/saved_discriminator.ckpt artefacts/saved_discriminator.ckpt
