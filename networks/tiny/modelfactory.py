class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(sizes, dataset='tinyimagenet', args=None):

        if dataset == "tinyimagenet":

            channels = 160
            return [
                ('conv2d-nbias', [channels, 3, 3, 3, 2, 1], ''),
                ('relu', [True], ''),

                ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
                ('relu', [True], ''),

                ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
                ('relu', [True], ''),

                ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
                ('relu', [True], ''),

                ('flatten', [], ''),
                ('rep', [], ''),

                ('linear-nbias', [640, 16 * channels], ''),
                ('relu', [True], ''),

                ('linear-nbias', [640, 640], ''),
                ('relu', [True], ''),
                ('head-nbias', [sizes[-1], 640], '')
            ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
