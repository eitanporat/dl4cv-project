from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 10000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 64, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 200, help='the number of generated images for evaluation') #50000
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

# optimizer
flags.DEFINE_integer('optimizer_kernel_size', 1, help='Optimizer Neural Network kernel size')
flags.DEFINE_integer('optimizer_out_channels', 3, help='Optimizer Neural Network Out Channels (Intermediate representation)')
flags.DEFINE_integer('optimizer_time_steps', 10, help='Optimizer Neural Network Number Of Layers / Optimizer Timesteps')
flags.DEFINE_integer('save_every', 20, help='interval for saving a picture to see progress')

flags.DEFINE_string('model_checkpoint', './logs/DDPM_CIFAR10_EPS/ckpt.pt', help='model checkpoint')
flags.DEFINE_string('sampler_checkpoint', '', help='sampler checkpoint')
flags.DEFINE_string('time_embedding_checkpoint', '', help='sampler checkpoint')
flags.DEFINE_string('discriminator_checkpoint', '', help='discriminator checkpoint')

flags.DEFINE_integer('T_reduced', 10, help='T reduced')
flags.DEFINE_string('sampler_type', 'momentum', help='sampler type')
flags.DEFINE_string('file_dir', '', help='sampler type')
flags.DEFINE_bool('train_time_embedding', False, help='sampler type')
flags.DEFINE_string('optimizer_type', 'AdamW', help='sampler type')

flags.DEFINE_integer('critic_iters', 5, help='Critic Iterations')
flags.DEFINE_integer('generator_iters', 10000000, help='Generator Iterations')

flags.DEFINE_integer('step_every', 1, help='Update weights every n steps')

