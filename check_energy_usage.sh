nvidia-smi dmon -s p -d 0.1sec -f energy.out
awk '!/^#/ && NF {print $2}' energy.out > pwr_values.txt
awk '!/^#/ {print $2}' energy.out | sort -n | tail -1 > max_pwr.txt