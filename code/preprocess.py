import os
import sys
import json
import pickle
import h5py
import argparse

import nltk
import tqdm
from PIL import Image

from config import cfg, cfg_from_file
from utils import load_label_embeddings, get_labels_concepts_filename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='shapes_train.yml', type=str)
    args = parser.parse_args()
    return args

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', 'CLEVR_{}_questions.json'.format(split))) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer, question['question_family_index']))

    with open('../data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    # root = sys.argv[1]

    # word_dic, answer_dic = process_question(root, 'train')
    # process_question(root, 'val', word_dic, answer_dic)

    # with open('../data/dic.pkl', 'wb') as f:
    #     pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    labels_matrix, concepts, concept_words = load_label_embeddings(cfg, get_concept_words=True)
    print('Concept words:\n')
    for word in concept_words:
        print(word)
    with h5py.File(get_labels_concepts_filename(cfg), 'w') as h5f:
        h5f.create_dataset('labels_matrix', data=labels_matrix)
        h5f.create_dataset('concepts', data=concepts)
