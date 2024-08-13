#!/bin/bash

# set num processes
N=24

# set num GPUs
NG=1

# set to write
write_mode="w"

# input dir
lc_dir=$1
# outuput dir
res_dir="$lc_dir"/results
#res_dir="$lc_dir"/results_transormed_t0_tE
#res_dir="$lc_dir"/results_different_init_vals
#res_dir="$lc_dir"/results_fitted_init_vals
#res_dir="$lc_dir"/results_fixed_t0_tE
#res_dir="$lc_dir"/track_test_2
#res_dir="$lc_dir"/test_ordering
#res_dir="$lc_dir"/mem_test_results

# set if microlensing: True or False
microlensing=$2

# get number of all files to process
#nf=$(ls "$lc_dir"/mock_lpc_3110.parquet* | wc -l)
nf=$(find "$lc_dir" -maxdepth 1 -type f -name '*.parquet*' | wc -l)
echo "num files = "$nf

#echo $res_dir

process_one () {

    in_file=$1
    i=$2
    wm=$3
    j=$4

    #gpu_id=$((i%NG))
    gpu_id=0 # use this also for the CPU

    #echo $in_file
    filename=$(basename -- "$in_file")
    #echo $filename

    outf="$res_dir"/"$i"_results.csv
    #outf="$res_dir"/"$i"_results_muwe_true.csv
    #outf="$res_dir"/back_"$filename"

    #echo "$in_file" "$lc_dir" "$outf" $wm $gpu_id $microlensing
    #python3 minimize_one_example.py "$in_file" "$lc_dir" "$outf" $wm $gpu_id $microlensing
    nice -20 python3 minimize_one_example.py "$in_file" "$lc_dir" "$outf" $wm $gpu_id $microlensing
    #nice -20 python3 compare_tracks.py "$in_file" "$lc_dir" "$outf" $wm $gpu_id $microlensing

    if [[ $((j%50)) == 0 ]]
    then
        echo "****************"
        echo "Completed "$((j*100/nf))"%"
        echo "****************"
    fi

}


# process all files in the input dir
#for lc_file in "$lc_dir"/mock_lpc_3110.parquet*
for lc_file in "$lc_dir"/*.parquet*
do
    ((i=i%N)); ((i++==0)); ((j++==0))

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi


    if [[ $j -gt $N ]]
    then
        write_mode="a" # set to append from N on
    fi

    process_one $lc_file $i $write_mode $j &

done

wait

echo "All examples done."
