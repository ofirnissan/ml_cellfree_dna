


import pysam

# Open the BAM or SAM file
samfile = pysam.AlignmentFile("/mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/DNABERT/examples/regression/1000247.bam", "rb")  # Replace "your_file.bam" with your actual file name

# Iterate over the first 10 reads
for i, read in enumerate(samfile.fetch()):
    if i == 10:
        break
    print(f"Read {i+1}: {read.query_sequence}")

# Close the file
samfile.close()