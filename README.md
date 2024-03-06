# HSC_selection_SimCLR

Backup of the codes in astro-node to obtain the HSC cutouts, preprocess the images, create the tensors, train the contrastive learning algorithm and visualize the latent space with UMAP.

The main contrastive learning code was develop on the basis of the [keras tutorial](https://keras.io/examples/vision/semisupervised_simclr/).

Line to run HSC cutout file:

python downloadCutout.py  --sw=6arcsec --sh=6arcsec --list=HSC_all_QSOs.txt --user=username --password=password --filter="HSC-G" --name="cutout_{tract}_{ra}_{dec}_{filter}"

