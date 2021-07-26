#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 10:01:19 2020

RICH (DI,TRI)-PHONE TEXT SELECTION ALGORITHM

@author: ivan
"""

import os
import re
import sys

digraphs = {'C_H': 'CH', 'D_Ź': 'DŹ'}
mode = 'uasr'
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
    'Z': 'Z'
}

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
                # print('WARNING:', rule1, rule2, excp)
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


def read_audio_sentences(f, inventory=None):
    txt = []
    audio_id = []
    try:
        with open(os.path.join(os.getcwd(), f), 'r', encoding='utf8') as corpus:
            inv = set(inventory)
            for line in corpus:
                tline = line.split('\t')
                if len(tline) == 2:
                    aid = tline[0]
                    tline = tline[1]
                else:
                    aid = ''
                    tline = tline[0]

                # tline = re.sub(ur'[^\P{P}\.]+', '', tline).upper().strip()
                tline = re.sub(r'[^\w\s\d]', '', tline).upper().strip()
                graphemes = '_'.join(tline)
                for d in digraphs:
                    graphemes = graphemes.replace(d, digraphs[d])
                graphemes = graphemes.strip().split('_')
                if inventory:
                    linv = set(graphemes)
                    linv.discard(' ')
                    if linv.issubset(inv):
                        txt.append(tline)
                        audio_id.append(aid.strip())
                    else:
                        print(linv.difference(inv), tline)
                else:
                    txt.append(tline)
                    audio_id.append(aid.strip())

    except FileNotFoundError:
        print('Error opening file:', f)

    return audio_id, txt


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} corpus.txt inventory.txt')
        raise SystemExit()

    if not os.path.exists('lab'):
        os.mkdir('lab')

    pinv, pmap = read_inventory(sys.argv[2])
    ids, corp = read_audio_sentences(sys.argv[1], pinv)

    text = ' '.join(corp)
    vocab = sorted(list(set(text.split())))
    ofn = sys.argv[1].split('.')[0]
    with open(os.path.join(os.getcwd(), ofn + '.vocab'), 'w', encoding='utf8') as vfile, \
            open(os.path.join(os.getcwd(), ofn + '.lex'), 'w', encoding='utf8') as lfile:
        for v in vocab:
            pron = phonemize(v, pmap)
            fsg_pron = []
            for phm in pron:
                if uasr_map.get(phm) is not None:
                    fsg_pron.append(uasr_map[phm])
                else:
                    fsg_pron.append(phm)
            vfile.write(v + '\n')
            lfile.write(v + '\t' + ' '.join(fsg_pron) + '\n')

    if len(list(set(ids))) == len(ids):
        with open(os.path.join(os.getcwd(), ofn + '.corpus'), 'w', encoding='utf8') as cfile:
            for id, c in zip(ids, corp):
                cfile.write(c + '\n')
                subfldr = 'lab/' + id.split('_')[0] + '-' + id.split('_')[1]
                ssfldr = subfldr + '/RECS/'

                if not os.path.exists(subfldr):
                    os.mkdir(subfldr)
                    os.mkdir(ssfldr)

                with open(os.path.join(os.getcwd(), subfldr, id + '.lab'), 'w', encoding='utf8') as labfile2:
                    labfile2.write('\n'.join(c.split()) + '\n')

print('Done')
