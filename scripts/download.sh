mkdir -p data
cd data

for url in 1_Z6LJXdoC16SwiIJ03c-yVU42jDvixjO 1IAnSJt2-NhWtOMugM7np0emBqs2IrYsj 1pJBKvOsNjPFTiy0-dm-pHDxmPhOuZxtJ  # training label, trainging image, test image
do 
    gdown ${url}
done

for f in test_image.zip train2017.zip training_label.zip
do 
    unzip -q ${f}
done

ln -s train2017 val2017

gdown 1AwUn5EebmmLBo7njjW_Ng1q9zDrqkNbB # checkpoint0033_4scale.pth

cd ..
