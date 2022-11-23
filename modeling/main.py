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
    d = 'gs://small-storage1/checkpoints/Diffs-v0/11-mix10-medium-tposaft-lr50/000300000/'
    sample_chain3 = "Hello world,"
    s = " I'm not sure what to do."
    with torch.inference_mode():
        device = 'cuda'
        model = CodifyModel.from_pretrained(d)
        model = model.to(device).to(torch.half).eval()
        config = model.config
        encoding = config.encoding
        test_s = sample_chain3 + s * 3
        test_t = encoding.encode(test_s)
        test_t = torch.tensor(test_t, device=device).unsqueeze(0)
        output = model.generate(test_t,
                                max_length=len(test_t) + 200,
                                temperature=0.8,
                                use_cache=True)
        print(encoding.decode(output[0].cpu().numpy()))
        i = 0