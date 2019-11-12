IFOLDER="/home/rodrigo/Downloads/sidtermo-test/input"
OFOLDER="/home/rodrigo/Downloads/sidtermo-test/output"
CSV_FILE="/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv"
BODY_PLANE=2
FIRST_SLICE="52"
TOTAL_SLICES="20"


python ./get_sidtermo_files.py -i $IFOLDER -o $OFOLDER -c $CSV_FILE -f $FIRST_SLICE -t $TOTAL_SLICES -b $BODY_PLANE -v
