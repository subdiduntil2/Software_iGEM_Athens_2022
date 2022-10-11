import sys
import numpy as np
sys.path.append("/usr/local/lib/python3.6/site-packages/RNA")
import _RNA as RNA
from Bio import SeqIO
import csv
records_16=list(SeqIO.parse("Vienna_RNA_16s_bef.fas", "fasta"))
records_22=list(SeqIO.parse("Vienna_RNA_22s_bef.fas", "fasta"))
names_16=[]
names_22=[]
seqs_16=[]
seqs_22=[]
bpps_16=[]
bpps_22=[]
for i in range(len(records_16)):
	nm_16=records_16[i].name
	nm_22=records_22[i].name
	seq_16=records_16[i].seq
	seq_22=records_22[i].seq
	(propensity_16,ee_16) = RNA.pf_fold(str(seq_16))
	(propensity_22,ee_22) = RNA.pf_fold(str(seq_22))
	names_16.append(nm_16)
	names_22.append(nm_22)
	seqs_16.append(seq_16)
	seqs_22.append(seq_22)
	bpps_16.append(abs(ee_16))
	bpps_22.append(abs(ee_22))


max_bpp_16=max(bpps_16)
max_bpp_22=max(bpps_22)
print(max_bpp_16,max_bpp_22)
bpps_16_norm=np.array(bpps_16)/max_bpp_16
bpps_22_norm=np.array(bpps_22)/max_bpp_22


bpps_16.insert(0,'Average bpp')
names_16.insert(0,'Seq Names')
seqs_16.insert(0,'Sequences')

bpps_22.insert(0,'Average bpp')
names_22.insert(0,'Seq Names')
seqs_22.insert(0,'Sequences')


rows_16 = zip(names_16,seqs_16,bpps_16)
rows_22 = zip(names_22,seqs_22,bpps_22)
print(rows_22)

with open("Vienna_RNA_16s_after.csv", "w") as f: 
	writer = csv.writer(f)
	for row_16 in rows_16:
		writer.writerow(row_16)

with open("Vienna_RNA_22s_after.csv", "w") as f: 
	writer = csv.writer(f)
	for row_22 in rows_22:
		writer.writerow(row_22)



	
	
	
