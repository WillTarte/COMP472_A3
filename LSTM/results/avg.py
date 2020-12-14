# yes this is pretty bad code

acc = []
yes_prec = []
no_prec = []
yes_recall = []
no_recall = []
yes_f1 = []
no_f1 = []

for i in range(1,11):
    with open(f'data/{i}eval_lstm.txt') as file:
        lines = file.readlines()
        acc.append(float(lines[0][:-1]))
        prec = lines[1].split(' ')
        yes_prec.append(float(prec[0]))
        no_prec.append(float(prec[2]))

        recall = lines[2].split(' ')
        yes_recall.append(float(recall[0]))
        no_recall.append(float(recall[2]))
        
        f1 = lines[3].split(' ')
        yes_f1.append(float(f1[0]))
        no_f1.append(float(f1[2]))



print(acc)
print(yes_prec)
print(no_prec)
print(yes_recall)
print(no_recall)
print(yes_f1)
print(no_f1)

import numpy as np

with open('avg_eval.txt', 'w') as out_file:
    out_file.write(f'{np.mean(acc)}\n')
    out_file.write(f'{np.mean(yes_prec)} {np.mean(no_prec)}\n')
    out_file.write(f'{np.mean(yes_recall)} {np.mean(no_recall)}\n')
    out_file.write(f'{np.mean(yes_f1)} {np.mean(no_f1)}\n')
