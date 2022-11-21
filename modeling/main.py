from modeling.codify_model import CodifyModel
import torch
_models = {
    'codify-medium': 'https://hug.....'
}

if __name__ == '__main__':
    # url TODO
    # model = CodifyModel.from_pretrained(_models['codify-medium'])
    # gs is working
    # model = CodifyModel.from_pretrained('gs://small-storage1/checkpoints/Diffs-v0/11-mix10-medium-tposaft-lr50/000300000/')
    # directory
    d = '/tmp/small-cache-container/small-storage1/checkpoints/Diffs-v0/11-mix10-medium-tposaft-lr50/000300000'
    with torch.inference_mode():
        model = CodifyModel.from_pretrained(d)
        model = model.to('cuda').to(torch.half).eval()
        i = 0