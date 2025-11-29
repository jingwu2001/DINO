mkdir -p data
cd data

# for url in 1_Z6LJXdoC16SwiIJ03c-yVU42jDvixjO 1IAnSJt2-NhWtOMugM7np0emBqs2IrYsj 1pJBKvOsNjPFTiy0-dm-pHDxmPhOuZxtJ  # training label, test image, training image
# do 
#     gdown ${url}
# done

# for f in test_image.zip train2017.zip training_label.zip
# do 
#     unzip -q ${f}
# done
# mv training_image train2017
# ln -s train2017 val2017


for url in 1IAnSJt2-NhWtOMugM7np0emBqs2IrYsj 15WD2-haLeGe2JQeVe4iS4TUwoF6R-waO 1rwxLKn4MAy0pUYKEWwvBy8-v2XazD66S # test_image train13, val15
do 
    gdown ${url}
done

for f in test_image.zip train13.zip val15.zip
do 
    unzip -q ${f}
done

mv train13 train2017
mv val15 val2017

cd ..


mkdir -p ckpt
cd ckpt

for url in 1AwUn5EebmmLBo7njjW_Ng1q9zDrqkNbB # 1Slrekf60spGumTi2wh49Aww5iEmgXZEg 1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_ 1DEoJzk8NL1hHQqTtDSkSHj5K6PRJ8bXq #checkpoint0033_4scale.pth, checkpoint0031_5scale.pth, checkpoint0011_4scale.pth, checkpoint0011_5scale.pth
do 
    gdown ${url}
done

cd ..
