from configs.base_options import BaseOptions

class InferenceOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)
        parser.add_argument('--result_dir', type=str, default='./results',
                            help='save result images into result_dir/exp_name')
        parser.add_argument('--ckpt_dir',   type=str,
                            default='./code/models/best_model_cosine_adamw.ckpt',
                            help='load ckpt path')
        parser.add_argument('--background_color', type=tuple, default=(1.0, 1.0, 1.0),
                            help='background color for cropping')
        
        return parser