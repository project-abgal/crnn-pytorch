"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [-l LM] [-w LMW] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]
-v VOCAB  vocabrary file path
-l LM     LM checkpoint path
-w LMW    LM weights
"""
from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import common_config as config
from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
from language_model import LitLstmLM
import json

def predict(crnn, dataloader, label2char, decode_method, beam_size, language_model, language_model_weight):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char,
                               language_model=language_model, language_model_weight=language_model_weight)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    arguments = docopt(__doc__)

    images = arguments['IMAGE']
    reload_checkpoint = arguments['-m']
    batch_size = int(arguments['-s'])
    decode_method = arguments['-d']
    beam_size = int(arguments['-b'])
    vocab_file_path = arguments['-v']
    language_model_path = arguments['-l']
    language_model_weight = float(arguments['-w'])

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    predict_dataset = Synth90kDataset(paths=images,
                                      img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    with open(vocab_file_path) as fi:
        vocab = json.load(fi)
    language_model = LitLstmLM.load_from_checkpoint(language_model_path, vocab=vocab)
    language_model.to(device)
    preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size,
                    language_model=language_model,
                    language_model_weight=language_model_weight)

    show_result(images, preds)


if __name__ == '__main__':
    main()
