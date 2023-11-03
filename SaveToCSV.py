import csv  

header = ['WindowSize', 'Accuracy','Macro_Average', 'F1_Score_0','F1_Score_1']

with open('Outputs.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
