import os


def prepare_dirs(experiment_name, out_dir, resume):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    tensorboard_dir = os.path.join(
            os.path.expanduser(out_dir), 'tensorboard', experiment_name)
    checkpoints_dir = os.path.join(
            os.path.expanduser(out_dir), 'checkpoints', experiment_name)

    if not resume:
        for i in range(1000):
            num_tbrd_dir = tensorboard_dir + '-{:03d}'.format(i)
            num_ckpt_dir = checkpoints_dir + '-{:03d}'.format(i)
            if not os.path.isdir(num_tbrd_dir) and \
               not os.path.isdir(num_ckpt_dir):
                break
        if i == 1000:
            raise NameError(
                    'There are 999 experiments with the same name already.'
                    ' Please use another name for your experiments.')
    else:
        num_tbrd_dir = tensorboard_dir
        num_ckpt_dir = checkpoints_dir

    if not os.path.isdir(num_tbrd_dir):
        os.makedirs(num_tbrd_dir)
    if not os.path.isdir(num_ckpt_dir):
        os.makedirs(num_ckpt_dir)
    return num_tbrd_dir, num_ckpt_dir
