#!/bin/bash

# bulk rename generated images
rename.ul synthesized_image gan_none gan_none/*.jpg
rename.ul synthesized_image gan_infection gan_infection/*.jpg
rename.ul synthesized_image gan_ischaemia gan_ischaemia/*.jpg
rename.ul synthesized_image gan_both gan_both/*.jpg

# generate label snippets
ls --literal gan_none/ > gan_none.csv
ls --literal gan_infection/ > gan_infection.csv
ls --literal gan_ischaemia/ > gan_ischaemia.csv
ls --literal gan_both/ > gan_both.csv
sed -i 's/.jpg/.jpg,1,0,0,0/g' gan_none.csv
sed -i 's/.jpg/.jpg,0,1,0,0/g' gan_infection.csv
sed -i 's/.jpg/.jpg,0,0,1,0/g' gan_ischaemia.csv
sed -i 's/.jpg/.jpg,0,0,0,1/g' gan_both.csv

# fuse all gan snippets together with pseudo-label extended training labels
cat gan_none.csv gan_infection.csv gan_ischaemia.csv gan_both.csv > gan_all.csv
cat all.csv gan_all.csv > train_pseudo_gan.csv
