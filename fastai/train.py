# fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import CSVLogger, SaveModelCallback

if __name__ == "__main__":
    # dataset paths
    data_path = '../dataset/cat_and_dog'

    # check cuda
    print(f'PyTorch version {torch.version.__version__}')
    print(f'CUDA is {torch.cuda.is_available()}')
    print(f'CuDnn is {torch.backends.cudnn.enabled}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define batch size
    bs = 8

    # data augmentation
    tfms = get_transforms(do_flip=True, flip_vert=False,
                          max_rotate=3., max_zoom=1.05,
                          max_lighting=0.1, max_warp=0.,
                          p_affine=0.5, p_lighting=0.5)

    # define databunch
    data = ImageDataBunch.from_folder(data_path, train="train", valid="valid",
                                      ds_tfms=tfms, size=bs, num_workers=1)
    data.normalize(imagenet_stats)

    model_name = 'cat_dog_fastai'
    # define base architecture Densenet121 and callback
    # (auto save model when valid_loss is improved and auto write performance result to csv)
    learn = cnn_learner(data=data, base_arch=models.densenet121,
                        callback_fns=[partial(CSVLogger, filename=f'logs_{model_name}'),
                                      partial(SaveModelCallback, monitor='valid_loss', name=f'{model_name}')]).to_fp16()

    # loss
    learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy)

    # optimizer
    learn.opt_func = partial(optim.Adam, betas=(0.9, 0.99))

    # metrics
    learn.metrics = [accuracy, AUROC()]

    # learning rate range
    lrs = learn.lr_range(slice(1e-4, 1e-3))

    # fit model (train only top layer for 1 epoch)
    learn.fit_one_cycle(cyc_len=1, max_lr=lrs)

    # unfreeze all layer
    learn.unfreeze()

    # fit model (train all layers for 10 epochs)
    learn.fit_one_cycle(cyc_len=10, max_lr=lrs)
