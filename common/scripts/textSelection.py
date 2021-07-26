#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 10:01:19 2020

RICH (DI,TRI)-PHONE TEXT SELECTION ALGORITHM

@author: ivan
"""

import os
import pickle
import random as rn
import re
import sys

import numpy as np

# Configuration parameters

mode = 'uasr'  # or "sampa phoneme inventory"
split = 1
offset = 0  # id starting number
num_sentences = 300
names = ["phones", "diphones", "triphones"]
scoring_type = 3  # 1, 2 or 3; choose 2 for negative log-prob weights, 3 for sentence weights
basic_type = 'diphones'

log = open("textSelection.log", "a")
sys.stdout = log

selected_sentence_ids = set()
delimiter = '$'
digraphs = {'C_H': 'CH', 'D_Ź': 'DŹ'}
exceptions = {
    '#C_W_$': '*',
    '#C_Ł_$': '*',
    '$_CH_': 'k',
    '$_CH_C': 'x',
    '$_H_#V': 'h',
    '$_W_#C': '*',
    '$_W_J': 'U v',
    '$_Ł_#C': '*',
    'A_Š_Ł': 'j S',
    'E_CH_': 'C',
    'I_CH_': 'C',
    'I_J_$': '*',
    'K_Ń_$': '*',
    'S_Ń_$': '*',
    'T_Ř_I': 't s',
    'T_Ř_Ě': 't s',
    'U_Š_Ł': 'j S',
    '_B_$': 'p',
    '_D_$': 't',
    '_DŹ_$': 't S',
    '_E_DŹ': 'e:',
    '_E_J': 'e:',
    '_E_Ć': 'e:',
    '_E_Č': 'e:',
    '_E_Ń': 'e:',
    '_E_Ž': 'e:',
    '_H_#C': '*',
    '_H_$': '*',
    '_N_K': 'n g',
    '_Ž_$': 'S',
    'Ě_CH_': 'C'
}

uasr_map = {
    'Uv': 'U v',
    'ts': 't s',
    'tS': 't S',
    'jn': 'j n',
    'dZ': 'd S',
    'dS': 'd S',
    'ng': 'n g',
    'Z': 'S'
}


def partition (list_in, n):
    rn.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def token_counts(corp, token_type, token_ids):
    corpus_size = len(corp)
    dataset = np.zeros((corpus_size, len(token_ids)))
    i = 0
    for phonlist in corp:
        if token_type == "phones":
            tokens = phonlist
        elif token_type == "diphones":
            tokens = ["{}_{}".format(phonlist[i], phonlist[i + 1]) for i in range(len(phonlist) - 1)]
        elif token_type == "triphones":
            tokens = ["{}^{}+{}".format(phonlist[i], phonlist[i + 1], phonlist[i + 2]) for i in range(len(phonlist) - 2)]
        else:
            raise SystemExit('Token type not defined')

        for token in tokens:
            idx = np.where(np.array(token_ids) == np.array(token))
            try:
                dataset[i][idx] += 1
            except:
                dataset[i][idx] = 1
        i += 1

    return dataset


def score_dataset(dataset, scorestats):
    scores = []
    for sentence in dataset:
        if scoring_type == 1 or scoring_type == 2:
            scores.append(np.dot(sentence, scorestats))
        elif scoring_type == 3:
            # According to Berry & Fadiga "Data-driven Design of a Sentence List for an Articulatory Speech Corpus"
            T = np.sum(sentence) # number of tokens (phonemes, di-tri phones)
            if T != 0:
                u = np.count_nonzero(np.array(sentence)) # unique tokens
                binsent = sentence
                binsent[np.nonzero(sentence)] = 1
                scr = 1 / T * np.dot(binsent, scorestats) * (u / T)
            else:
                scr = 0
            scores.append(scr)
        else:
            raise SystemExit('Wrong scoring type.')

    best_sentence_id = np.where(scores == np.nanmax(scores))[0][0]
    selected_sentence_ids.add(best_sentence_id)
    return best_sentence_id


def check_exceptions(rules):
    phns = []
    for e in exceptions:
        for rule in rules:
            if e in rule:
                phns.extend([exceptions[e]])
    return phns


def phonemize(word, phmap):
    phns = []
    if phmap is not None:
        graphemes = '_'.join(list('$' + word.strip() + '$'))
        for d in digraphs:
            graphemes = graphemes.replace(d, digraphs[d])
        graphemes = graphemes.split('_')
        for g in range(len(graphemes) - 2):
            left = graphemes[g]
            grapheme = graphemes[g + 1]
            right = graphemes[g + 2]
            rule1 = left + '_' + grapheme + '_' + right
            rule2 = ('#' + phmap[left][1] if left != '$' else '$') + '_' + grapheme + '_' + (
                '#' + phmap[right][1] if right != '$' else '$')
            excp = list(set(check_exceptions([rule1, rule2])))
            # print(excp, len(excp))
            if len(excp) > 1:
                #print('WARNING:', rule1, rule2, excp)
                excp = excp[0]
            phn = []
            if len(excp) == 1:
                # print(rule1, rule2, excp)
                if excp[0] != '*':
                    phn = excp[0].split(' ')
            else:
                phn = phmap[grapheme][0].split(' ')
            phns.extend(phn)
        return phns
    else:
        return word.strip()


def read_inventory(f):
    try:
        with open(os.path.join(os.getcwd(), f), 'r', encoding='utf8') as invfile:
            pmap = {}
            phn = []
            for line in invfile:
                p = line.strip().split('\t')
                pmap[p[0]] = (p[1], p[2])
                phn.append(p[0])
                # print(p)
    except FileNotFoundError:
        print('Error opening file:', f['fn'])

    return phn, pmap


def read_sentences(f, inventory=None):
    try:
        with open(os.path.join(os.getcwd(), f), 'r', encoding='utf8') as corpus:
            txt = []
            sents = []
            audio_id = []
            inv = set(inventory)
            for line in corpus:
                parts = line.split('\t')
                if len(parts) == 2:
                    aid = parts[0]
                    tline = parts[1]
                else:
                    aid = ''
                    tline = parts[0]
                tline_p = re.sub(r'[^\w\s]', '', tline).upper().strip()
                graphemes = '_'.join(tline_p)
                for d in digraphs:
                    graphemes = graphemes.replace(d, digraphs[d])
                graphemes = graphemes.strip().split('_')
                if inventory:
                    linv = set(graphemes)
                    linv.discard(' ')
                    if linv.issubset(inv):
                        txt.append(tline)
                        sents.append(tline_p)
                        audio_id.append(aid.strip())
                    else:
                        print(linv.difference(inv), tline)
                else:
                    txt.append(tline)
                    sents.append(tline_p)
                    audio_id.append(aid.strip())

    except FileNotFoundError:
        print('Error opening file:', f)

    return audio_id, sents, txt


def read_lexicon(flex):
    ret = {}
    lex = None
    if flex == 'None':
        return ret
    for fl in flex.split(' '):
        fn = os.path.join(fl)
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                lex = f.readlines()
            lex = [l.strip().split('\t') for l in lex]
            lex = {l[0]: [p.split(' ') for p in l[1:]] for l in lex}
            ret.update(lex)
        except FileNotFoundError:
            print('Error opening file:', fn)
    return ret


def map_prons(sent, lex, pmap, aids=None, orig=None):
    pronunciation = []
    sents = []
    auds = []
    orgsent = []
    lexset = set(lex.keys())
    for idx, s in enumerate(sent):
        prn = []
        words = s.strip().split(' ')
        if not set(words).issubset(lexset) and bool(lex):
            continue
        prn.extend('.')
        phn=[]
        for w in words:
            if not lex.get(w):
                phn = phonemize(w, pmap)
            else:
                phn = lex[w][0]

            prn.extend(phn)
            prn.extend('.')

        if mode == 'uasr':
            map_temp = []
            for n in prn:
                uasr_phn = uasr_map.get(n, n).split()
                for u in uasr_phn:
                    map_temp.append(u)
            prn = map_temp

        pronunciation.append(prn)
        sents.append(s)
        if aids:
            auds.append(audio[idx])
        if orig:
            orgsent.append(orig[idx])

    return pronunciation, sents, auds, orgsent


def get_stats(p):
    stats = {}
    phones = {}
    diphones = {}
    triphones = {}

    for ln in p:
        phonelist = ln
        for phon in phonelist:
            try:
                phones[phon] += 1
            except:
                phones[phon] = 1

        # extract diphones from phones
        for i in range(len(phonelist) - 1):
            dip = "{}_{}".format(phonelist[i], phonelist[i + 1])
            # if dip in diphones:
            try:
                diphones[dip] += 1
            except:
                diphones[dip] = 1

        for i in range(len(phonelist) - 2):
            tri = "{}^{}+{}".format(phonelist[i], phonelist[i + 1], phonelist[i + 2])
            # if tri in triphones:
            try:
                triphones[tri] += 1
            except:
                triphones[tri] = 1

    stats['phones'] = phones
    stats['diphones'] = diphones
    stats['triphones'] = triphones

    return stats


# get wishlist
def scorelists(stat, name):
    tokens = list(stat[name].keys())
    # Score 1: each type shoul be present at least once
    score1 = np.full(np.array(list(stat[name].values())).shape[0], 1)

    # Score 2: each base type according the negative probability logs
    score2 = np.array(list(stat[name].values()))
    score2 = -np.log(score2 / np.sum(score2))

    return tokens, list(score1), list(score2), list(score2)


def sent_selection(stats, base_type='phones', score_type=1):
    if 1 > score_type > 3:
        raise SystemExit("Wrong scoring type")

    scores = scorelists(stats, base_type)
    dataset = token_counts(utterances, base_type, scores[0])
    score = scores[score_type]

    for it in range(num_sentences):
        best_sentence_id = score_dataset(dataset, score)
        # print(best_sentence_id)
        #sentsel.write(original[best_sentence_id])
        #uttssel.write(' '.join(utterances[best_sentence_id]) + '\n')
        #audsel.write(audio[best_sentence_id] + '\n')
        dataset[best_sentence_id] = np.zeros(dataset.shape[-1])


if __name__ == "__main__":

    if not os.path.exists('sent'):
        os.mkdir('sent')

    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} corpus.txt vocab.txt sentences.txt lexicon.txt [phninv.txt]')
        raise SystemExit()

    if len(sys.argv) == 6:
        phoninv, phonmap = read_inventory(sys.argv[5])
    else:
        phoninv = None
        phonmap = None

    if not os.path.exists('stats.pickle'):
        # Big corpus (wiki dump or sorbian institute monolinugual)
        _, corpus, _ = read_sentences(sys.argv[1], phoninv)
        vocab = read_lexicon(sys.argv[2])
        prons, corpus, _, _ = map_prons(corpus, vocab, phonmap, None, None)
        ideal_stats = get_stats(prons)
        with open('stats.pickle', 'wb') as statspkl:
            pickle.dump(ideal_stats, statspkl, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('stats.pickle', 'rb') as statspkl:
            ideal_stats = pickle.load(statspkl)

    # Custom corpus (common voice)
    audio, sentences, original = read_sentences(sys.argv[3], phoninv)
    lexicon = read_lexicon(sys.argv[4])
    utterances, sentences, audio, original = map_prons(sentences, lexicon, phonmap, audio, original)
    phnstats = ideal_stats
    sent_selection(phnstats, basic_type, scoring_type)

    # split the set
    split_ids = partition(list(selected_sentence_ids), split)

    for i, idx in enumerate(split_ids):
        out_file = './sent/sentsel_' + basic_type + '_' + str(scoring_type) + '_' + mode + '_' + str(i+1) + '.txt'
        utts_out = './sent/uttssel_' + basic_type + '_' + str(scoring_type) + '_' + mode + '_' + str(i+1) + '.txt'
        a_ids = './sent/audio_' + basic_type + '_' + str(scoring_type) + '_' + mode + '_' + str(i+1) + '.txt'
        sentsel = open(os.path.join(os.getcwd(), out_file), "w", encoding='utf-8')
        uttssel = open(os.path.join(os.getcwd(), utts_out), "w", encoding='utf-8')
        audsel = open(os.path.join(os.getcwd(), a_ids), "w", encoding='utf-8')

        for k, sm_id in enumerate(idx):
            header = 'HSB_' + str(i+1) + '_' + str(k+offset).zfill(3) + '\t'
            sentsel.write(header + original[sm_id])
            #labfile = open(os.path.join(os.getcwd(), 'lab/' + audio[sm_id].split('.')[0]+'.lab'), "w", encoding='utf-8')
            #labfile.write(sentences[sm_id] + '\n')
            #labfile.write(' '.join(utterances[sm_id]) + '\n')
            #labfile.close()
            uttssel.write(header + ' '.join(utterances[sm_id]) + '\n')
            audsel.write(audio[sm_id].split('.')[0] + '\n')

        utt_sel = list(np.array(utterances)[idx])
        sent_sel = list(np.array(sentences)[idx])
        audio_sel = list(np.array(audio)[idx])

        wrd_cnt = len((' '.join([''.join(s) for s in sent_sel])).split(' '))
        char_cnt = len((' '.join([' '.join(u) for u in utt_sel]).split(' ')))

        sel_stats = get_stats(utt_sel)
        whtspc = sel_stats['phones']['.']

        rnd_sent = rn.sample(utterances, int(num_sentences/split))
        rnd_stats = get_stats(rnd_sent)

        print(f'\n\nSet Nr.: {i+1}')
        print(f'sentences: {int(num_sentences/split)}')
        print(f'sell word count: {wrd_cnt}')
        # Fonagy, I.; K. Magdics (1960). "Speed of utterance in phrases of different length".
        # Language and Speech. 3 (4): 179–192. doi:10.1177/002383096000300401.
        pps_min = 10 #9.4  # reading poetry
        pps_max = 15 #13.83  # commenting sport

        print(f'estimated duration by phoneme units (min): {((char_cnt-whtspc) / pps_max) / 60} - {((char_cnt-whtspc) / pps_min) / 60}')
        # http://prosodia.upf.edu/home/arxiu/publicacions/rodero/
        # rodero_a-comparative-analysis-of-speech-rate-and-perception-in-radio-bulletins.pdf
        wpm_min = 100 #168  # Englisch BBC
        wpm_max = 160 #210  # Spanish RNE
        print(f'estimated duration by number of words (min): {(wrd_cnt / wpm_max)} - {(wrd_cnt / wpm_min)}')

        for b_type in names:
            total_tokens = list(phnstats[b_type].keys())
            sel_tokens = list(sel_stats[b_type].keys())
            rnd_tokens = list(rnd_stats[b_type].keys())
            print(f'\ntype: {b_type}')
            print(f'total tokens: {len(total_tokens)}')
            print(f'random ratio: {len(rnd_tokens) / len(total_tokens)}')
            print(f'rand diff: {set(total_tokens).difference(set(sel_tokens))}')
            print(f'selected ratio: {len(sel_tokens) / len(total_tokens)}')
            print(f'sell diff: {set(total_tokens).difference(set(sel_tokens))}')

    print('Done')
